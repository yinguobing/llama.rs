// An implementation of LLaMA 3 https://llama.meta.com/llama3
//
// Based on candle examples
// https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama

use candle_core::{utils, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::llama};
use tokenizers::Tokenizer;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model_path: &str = "/home/robin/hdd/Meta-Llama-3-8B-Instruct";
    println!("loading the model weights from {model_path}");

    // Loading model configuration
    let config_filename = format!("{model_path}/config.json");
    let config: llama::LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let use_flash_attn: bool = true;
    let config = config.into_config(use_flash_attn);

    // Check if GPU available
    let device = if utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    // Loading model weights
    let safetensors = vec![
        format!("{model_path}/model-00001-of-00004.safetensors"),
        format!("{model_path}/model-00002-of-00004.safetensors"),
        format!("{model_path}/model-00003-of-00004.safetensors"),
        format!("{model_path}/model-00004-of-00004.safetensors"),
    ];
    let dtype = DType::BF16;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &device)? };

    // Init model
    let llama = llama::Llama::load(vb, &config)?;

    // Init cache
    let use_kv_cache: bool = true;
    let mut cache = llama::Cache::new(use_kv_cache, dtype, &config, &device)?;

    // Init tokenizer
    let tokenizer_filename = format!("{model_path}/tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| e.to_string())?;
    let eos_token: &str = "<|eot_id|>";
    let eos_token_id = tokenizer.token_to_id(eos_token);

    // Init sampler
    let temperature: f64 = 0.8;
    let top_p: f64 = 0.9;
    let seed: u64 = 299792458;
    let mut logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));

    // Init tokens
    let prompt: &str = r"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Who are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
    println!("{prompt}");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();

    println!("Inferring...");

    // The length of the sample to generate (in tokens).
    let sample_len: usize = 4096;

    // Penalty to be applied for repeating tokens, 1. means no penalty.
    let repeat_penalty: f32 = 1.1;

    // The context size to consider for the repeat penalty.
    let repeat_last_n: usize = 128;

    let mut current_index = 0;
    let mut token_generated = 0;
    let start_gen = std::time::Instant::now();
    for index in 0..sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, current_index)
        } else {
            (tokens.len(), 0)
        };
        let context = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(context, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;

        // Suppress repetition?
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        current_index += context.len();

        // Sample a token from the model output
        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // End of generation?
        if Some(next_token) == eos_token_id {
            break;
        }
    }

    // Decode the tokens
    let text = tokenizer.decode(&tokens, true).map_err(|e| e.to_string())?;
    print!("{text}");

    let dt = start_gen.elapsed();

    // Metrics
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
