export type ProviderType = 'openai' | 'deepseek' | 'custom'

export interface Message {
  id?: number
  role: 'user' | 'assistant'
  content: string
  audio_url?: string
  created_at?: string
}

export interface ChatResponse {
  text: string
  conversation?: {
    id: number
    url: string
    messages: {
      id: number
      role: 'user' | 'assistant'
      content: string
      audio_url?: string
      created_at: string
    }[]
  }
}

export interface ModelParams {
  stt: {
    beamSize: number
    temperature: number
  }
  tts: {
    speed: number
    pitch: number
  }
  api: {
    url: string
    providerType: ProviderType
    apiKey: string
    baseUrl?: string
    modelName?: string
  }
}

export type SliderValue = [number]
