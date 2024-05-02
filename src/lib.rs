// An implementation of LLaMA 3 https://llama.meta.com/llama3
//
// Based on candle examples
// https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

pub use candle_core::DType;
use candle_core::{utils, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::llama};
use tokenizers::Tokenizer;

#[derive(Error, Debug)]
pub enum LlamaError {
    #[error("Configuration error, {0}")]
    InvalidConfig(String),
    #[error("Tokenizer error, {0}")]
    TokenizerError(String),
    #[error("Infer error, {0}")]
    InferError(String),
    #[error("Prompt error, {0}")]
    PromptError(String),
}

/// Returns the best device to use for inference.
pub fn auto_device() -> Device {
    if utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}

#[derive(Debug)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

#[derive(Debug)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    fn encode(&self) -> String {
        match self.role {
            Role::System => {
                format!(
                    "<|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>",
                    self.content
                )
            }
            Role::User => {
                format!(
                    "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|>",
                    self.content
                )
            }
            Role::Assistant => {
                format!(
                    "<|start_header_id|>assistant<|end_header_id|>\n{}<|eot_id|>",
                    self.content
                )
            }
        }
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.role, self.content)
    }
}

#[derive(Debug, Clone)]
pub struct ChatConfig {
    pub use_flash_attn: bool,
    pub use_kv_cache: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
    pub device: Device,
    pub dtype: DType,
    pub eos_token: Option<String>,
    pub repeat_penalty: f32,
    pub repeat_last_n: u64,
    pub max_context_length: u64,
}

pub struct LlamaChat {
    config: ChatConfig,
    model: llama::Llama,
    cache: llama::Cache,
    logits_processor: LogitsProcessor,
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    eos_token_ids: HashSet<u32>,
    index: u64,
    current_index: usize,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

impl LlamaChat {
    pub fn new<P: AsRef<Path>>(model_path: P, config: &ChatConfig) -> Result<Self, LlamaError> {
        // We need directory rather than files
        let mut model_path: PathBuf = if model_path.as_ref().is_file() {
            model_path
                .as_ref()
                .parent()
                .ok_or(LlamaError::InvalidConfig(format!(
                    "Can not get parent directory of {:?}",
                    model_path.as_ref().to_str()
                )))?
                .to_path_buf()
        } else {
            model_path.as_ref().to_path_buf()
        };

        // Loading model configuration
        model_path.push("config.json");
        let model_config: llama::LlamaConfig = serde_json::from_str(
            &std::fs::read_to_string(&model_path)
                .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?,
        )
        .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?;
        let model_config = model_config.into_config(config.use_flash_attn);

        // Filter all safetensor files
        model_path.set_file_name("model.safetensors.index.json");
        let safetensors: SafetensorsIndex = serde_json::from_str(
            &std::fs::read_to_string(&model_path)
                .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?,
        )
        .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?;
        let mut safetensors_files = HashSet::new();
        for value in safetensors.weight_map.values() {
            safetensors_files.insert(value.to_string());
        }
        let safetensors_files = safetensors_files
            .iter()
            .map(|v| {
                model_path.set_file_name(v);
                model_path.clone()
            })
            .collect::<Vec<_>>();

        // Init model
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, config.dtype, &config.device)
                .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?
        };
        let model = llama::Llama::load(vb, &model_config)
            .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?;

        // Init cache
        let cache = llama::Cache::new(
            config.use_kv_cache,
            config.dtype,
            &model_config,
            &config.device,
        )
        .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?;

        // Init tokenizer
        model_path.set_file_name("tokenizer.json");
        let tokenizer = Tokenizer::from_file(model_path)
            .map_err(|e| LlamaError::InvalidConfig(e.to_string()))?;

        // Additional EOS token id?
        let mut eos_token_ids = HashSet::new();
        if let Some(id) = model_config.eos_token_id {
            eos_token_ids.insert(id);
        }
        let _ = config
            .eos_token
            .as_ref()
            .and_then(|t| tokenizer.token_to_id(t))
            .map(|t| eos_token_ids.insert(t));

        // Init sampler
        let logits_processor =
            LogitsProcessor::new(config.seed, Some(config.temperature), Some(config.top_p));

        Ok(Self {
            model,
            cache,
            logits_processor,
            tokenizer,
            tokens: vec![],
            eos_token_ids,
            config: config.to_owned(),
            index: 0,
            current_index: 0,
        })
    }

    pub fn encode(&self, msgs: &Vec<Message>) -> String {
        let mut encoded = String::new();
        encoded.push_str("<|begin_of_text|>");
        for msg in msgs {
            encoded.push_str(&msg.encode());
        }
        encoded.push_str("<|start_header_id|>assistant<|end_header_id|>\n");
        encoded
    }

    pub fn generate(&mut self, prompt: &str) -> Result<(), LlamaError> {
        self.tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| LlamaError::TokenizerError(e.to_string()))?
            .get_ids()
            .to_vec();
        self.index = 0;
        self.current_index = 0;
        Ok(())
    }
}

impl Iterator for &mut LlamaChat {
    type Item = Result<String, LlamaError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.config.max_context_length {
            return None;
        }

        let (context_size, context_index) = if self.cache.use_kv_cache && self.index > 0 {
            (1, self.current_index)
        } else {
            (self.tokens.len(), 0)
        };
        let context = &self.tokens[self.tokens.len().saturating_sub(context_size)..];
        let input = match Tensor::new(context, &self.config.device).and_then(|x| x.unsqueeze(0)) {
            Ok(x) => x,
            Err(e) => {
                return Some(Err(LlamaError::InferError(e.to_string())));
            }
        };

        let logits = match self
            .model
            .forward(&input, context_index, &mut self.cache)
            .and_then(|x| x.squeeze(0))
        {
            Ok(x) => x,
            Err(e) => {
                return Some(Err(LlamaError::InferError(e.to_string())));
            }
        };

        // Suppress repetition?
        let logits = if self.config.repeat_penalty == 1. {
            logits
        } else {
            let start_at = self
                .tokens
                .len()
                .saturating_sub(self.config.repeat_last_n as usize);
            match candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.config.repeat_penalty,
                &self.tokens[start_at..],
            ) {
                Ok(x) => x,
                Err(e) => {
                    return Some(Err(LlamaError::InferError(e.to_string())));
                }
            }
        };
        self.current_index += context.len();

        // Sample a token from the model output
        let next_token = match self.logits_processor.sample(&logits) {
            Ok(x) => x,
            Err(e) => {
                return Some(Err(LlamaError::InferError(e.to_string())));
            }
        };
        self.tokens.push(next_token);

        // End of generation?
        if self.eos_token_ids.contains(&next_token) {
            return None;
        }

        // Decode the tokens
        let result = match self.tokenizer.decode(&[next_token], true) {
            Ok(x) => x,
            Err(e) => {
                return Some(Err(LlamaError::InferError(e.to_string())));
            }
        };
        self.index += 1;

        Some(Ok(result))
    }
}
