export interface WebGLState {
  activeState: string;
  isLoaded: boolean;
  progress: number;
}
export interface AppConfig {
  debug: boolean;
  quality: "high" | "medium" | "low";
}
