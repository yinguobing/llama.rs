# llama3
Running LLaMA 3 with Rust.

- Chat with LLaMA 3 Instruct model
- Candle as backend, support CUDA acceleration
- Stream output

## Running
```sh
cargo run --release --features flash-attn
```

## Usage
```rust
use llama3::{auto_device, ChatConfig, DType, LlamaChat, Message, Role};
use std::io::Write;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Model weights on disk
    let model_path: &str = "/home/robin/hdd/Meta-Llama-3-8B-Instruct";

    // Setup config
    let config = ChatConfig {
        use_flash_attn: true,                      // Speedup
        use_kv_cache: true,                        // Speedup
        temperature: 0.8,                          // Higher means more diversity
        top_p: 0.9,                                // Narrow the sampling distribution
        seed: 299792458,                           // Seed for random process
        device: auto_device(),                     // Fetch available accelerate device
        dtype: DType::BF16,                        // FP16, FP32, etc.
        repeat_penalty: 1.1,                       // Penalty to repeating tokens
        repeat_last_n: 128,                        // The context size for the repeat penalty
        eos_token: Some("<|eot_id|>".to_string()), // Additional EOS token
        max_context_length: 8192,                  // Max length of the sample (in tokens).
    };

    // Init model
    println!("Loading the model weights from {model_path}");
    let mut model = LlamaChat::new(model_path, &config)?;

    // Init conversation
    let mut messages: Vec<Message> = vec![
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
    for m in messages.iter() {
        println!("{m}");
    }

    // Generation: Round 1
    model.generate(&prompt)?;
    let mut response = String::new();
    for r in &mut model {
        let s = r?;
        print!("{}", s);
        std::io::stdout().flush().unwrap();
        response.push_str(&s);
    }
    println!();

    // Update the conversation for the next round generation
    messages.push(Message {
        role: Role::Assistant,
        content: response,
    });
    messages.push(Message {
        role: Role::User,
        content: "My name is Robin! Tell me something about this name.".to_string(),
    });
    let prompt = model.encode(&messages);
    println!("{}", messages.last().unwrap());

    // Generation: Round 2
    model.generate(&prompt)?;
    let mut response = String::new();
    for r in &mut model {
        let s = r?;
        print!("{}", s);
        std::io::stdout().flush().unwrap();
        response.push_str(&s);
    }
    println!();

    Ok(())
}

```

## References
Prompt formats: https://llama.meta.com/docs/model-cards-and-prompt-formats/