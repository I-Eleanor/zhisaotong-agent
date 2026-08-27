/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端 API 基础地址，默认同源 "" */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}