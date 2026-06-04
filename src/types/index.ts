export interface LlamaServerPreset {
  id: string;
  name: string;
  model: string;
  host: string;
  port: number;
  ngl: number;
  ctx_size: number;
  threads: number;
  jinja: boolean;
  spec_draft_model: string | null;
  spec_draft_n_max: number | null;
}

export interface LlamaPerset {
  llamaServer: LlamaServerPreset[];
}

export interface AppConfig {
  llama_cpp_path: string;
  models_path: string;
  presets: LlamaPerset;
}

export interface ServerConfig {
  llama_cpp_path: string;
  model: string;
  host: string;
  port: number;
  ngl: number;
  ctx_size: number;
  threads: number;
  jinja: boolean;
  spec_draft_model: string | null;
  spec_draft_n_max: number | null;
}

export interface ModelInfo {
  name: string;
  path: string;
}

export interface BinaryInfo {
  name: string;
  path: string;
  exists: boolean;
  description: string;
}
