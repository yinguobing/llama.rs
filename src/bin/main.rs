// Chat with LLaMA 3
use llama::{auto_device, ChatConfig, DType, LlamaChat, Message, Role};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let model_path: &str = "/home/robin/hdd/Meta-Llama-3-8B-Instruct";
    println!("loading the model weights from {model_path}");

    // Fetch available accelerate device
    let device = auto_device();

    // Model configuration
    let dtype = DType::BF16;
    let use_flash_attn = true;
    let use_kv_cache = true;

    // The length of the sample to generate (in tokens).
    let max_context_length = 8192;

    // Penalty to be applied for repeating tokens, 1. means no penalty.
    let repeat_penalty: f32 = 1.1;

    // The context size to consider for the repeat penalty.
    let repeat_last_n: u64 = 128;

    // Override EOS token
    let eos_token = Some("<|eot_id|>".to_string());

    // Logits sampling parameters
    let temperature = 0.8;
    let top_p = 0.9;
    let seed = 299792458;

    // Setup config
    let config = ChatConfig {
        use_flash_attn,
        use_kv_cache,
        temperature,
        top_p,
        seed,
        device,
        dtype,
        repeat_penalty,
        repeat_last_n,
        eos_token,
        max_context_length,
    };

    // Init model
    let mut model = LlamaChat::new(model_path, &config)?;

    // Init conversation
    let messages: Vec<Message> = vec![
        Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Who are you?".to_string(),
        },
    ];
    let prompt = model.encode(&messages);
    println!("{prompt}");

    // Generating
    let start_gen = std::time::Instant::now();
    let response = model.generate(&prompt)?;
    let duration = start_gen.elapsed();
    println!("{response}");

    // Metrics
    println!("Done in {} seconds.", duration.as_secs());

    Ok(())
}
