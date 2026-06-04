use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlamaServerPreset {
    pub id: String,
    pub name: String,
    pub model: String,
    pub host: String,
    pub port: u16,
    pub ngl: u32,
    pub ctx_size: u32,
    pub threads: u32,
    pub jinja: bool,
    pub spec_draft_model: Option<String>,
    pub spec_draft_n_max: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LlamaPerset {
    pub llama_server: Vec<LlamaServerPreset>
}

impl Default for LlamaPerset {
    fn default() -> Self {
        Self {
            llama_server: Vec::new()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AppConfig {
    pub llama_cpp_path: String,
    pub models_path: String,
    #[serde(default)]
    pub presets: LlamaPerset
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            llama_cpp_path: String::new(),
            models_path: String::new(),
            presets: LlamaPerset::default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ServerConfig {
    pub llama_cpp_path: String,
    pub model: String,
    pub host: String,
    pub port: u16,
    pub ngl: u32,
    pub ctx_size: u32,
    pub threads: u32,
    pub jinja: bool,
    pub spec_draft_model: Option<String>,
    pub spec_draft_n_max: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryInfo {
    pub name: String,
    pub path: String,
    pub exists: bool,
    pub description: String,
}