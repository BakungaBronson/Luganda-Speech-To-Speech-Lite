import { ModelParams, ProviderType, SliderValue } from '../types'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  Label,
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
  Button,
  Slider
} from './ui'
import { Settings } from 'lucide-react'

interface SettingsPanelProps {
  modelParams: ModelParams
  onSettingsChange: (settings: ModelParams) => void
}

export function SettingsPanel({ modelParams, onSettingsChange }: SettingsPanelProps) {
  const handleSliderChange = (
    category: 'stt' | 'tts',
    param: 'beamSize' | 'temperature' | 'speed' | 'pitch',
    value: SliderValue
  ) => {
    onSettingsChange({
      ...modelParams,
      [category]: {
        ...modelParams[category],
        [param]: value[0]
      }
    })
  }

  const handleAPIChange = (apiConfig: Partial<ModelParams['api']>) => {
    onSettingsChange({
      ...modelParams,
      api: {
        ...modelParams.api,
        ...apiConfig
      }
    })
  }

  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Model Parameters</SheetTitle>
          <SheetDescription>
            Adjust the speech-to-text and text-to-speech parameters.
          </SheetDescription>
        </SheetHeader>
        <div className="grid gap-4 py-4">
          <div className="space-y-4">
            <h3 className="font-medium">API Settings</h3>
            <div className="grid gap-2">
              <Label htmlFor="api-url">API URL</Label>
              <input
                type="text"
                id="api-url"
                name="api-url"
                aria-label="API URL"
                placeholder="Enter API URL"
                value={modelParams.api.url}
                onChange={(e) => handleAPIChange({
                  url: e.target.value
                })}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </div>

            <div className="grid gap-2">
              <Label>Provider</Label>
              <Select
                value={modelParams.api.providerType}
                onValueChange={(value: ProviderType) => handleAPIChange({
                  providerType: value,
                  baseUrl: value === 'deepseek' ? 'https://api.deepseek.com' : '',
                  modelName: value === 'deepseek' ? 'deepseek-chat' : ''
                })}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="deepseek">DeepSeek</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="api-key">API Key</Label>
              <input
                type="password"
                id="api-key"
                name="api-key"
                aria-label="API Key"
                placeholder="Enter API Key"
                value={modelParams.api.apiKey}
                onChange={(e) => handleAPIChange({
                  apiKey: e.target.value
                })}
                className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
            </div>

            {modelParams.api.providerType === 'custom' && (
              <>
                <div className="grid gap-2">
                  <Label htmlFor="base-url">Base URL</Label>
                  <input
                    type="text"
                    id="base-url"
                    name="base-url"
                    aria-label="Base URL"
                    placeholder="e.g. https://api.example.com"
                    value={modelParams.api.baseUrl}
                    onChange={(e) => handleAPIChange({
                      baseUrl: e.target.value
                    })}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  />
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="model-name">Model Name</Label>
                  <input
                    type="text"
                    id="model-name"
                    name="model-name"
                    aria-label="Model Name"
                    placeholder="e.g. custom-model"
                    value={modelParams.api.modelName}
                    onChange={(e) => handleAPIChange({
                      modelName: e.target.value
                    })}
                    className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                  />
                </div>
              </>
            )}
            
            <div className="grid gap-2">
              <Label htmlFor="system-prompt">System Prompt</Label>
              <textarea
                id="system-prompt"
                name="system-prompt"
                aria-label="System Prompt"
                placeholder="Enter system prompt"
                value={modelParams.api.systemPrompt || "You are an expert Luganda language instructor with native-level fluency in both English and Luganda. You understand the nuances, idioms, and cultural context of Luganda communication. When users send Luganda text that may contain speech-to-text transcription errors or variations, you will intelligently interpret their meaning using context clues and respond naturally in proper Luganda. Your responses should maintain authentic Luganda grammar, tone, and cultural appropriateness while accommodating common speech pattern variations. If the input contains unclear words, you will match them to the most likely intended Luganda words based on pronunciation patterns and context. Engage in natural Luganda conversation while subtly modeling correct usage. Your outputs must be concise and not contain any jargon."}
                onChange={(e) => handleAPIChange({
                  systemPrompt: e.target.value
                })}
                className="flex min-h-[120px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
              />
              <Button
                variant="outline"
                className="w-full"
                onClick={() => handleAPIChange({
                  systemPrompt: "You are an expert Luganda language instructor with native-level fluency in both English and Luganda. You understand the nuances, idioms, and cultural context of Luganda communication. When users send Luganda text that may contain speech-to-text transcription errors or variations, you will intelligently interpret their meaning using context clues and respond naturally in proper Luganda. Your responses should maintain authentic Luganda grammar, tone, and cultural appropriateness while accommodating common speech pattern variations. If the input contains unclear words, you will match them to the most likely intended Luganda words based on pronunciation patterns and context. Engage in natural Luganda conversation while subtly modeling correct usage. Your outputs must be concise and not contain any jargon."
                })}
              >
                Reset to Default
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="font-medium">Speech-to-Text</h3>
            <div className="grid gap-2">
              <Label>Beam Size ({modelParams.stt.beamSize})</Label>
              <Slider
                value={[modelParams.stt.beamSize]}
                min={1}
                max={10}
                step={1}
                onValueChange={(value: SliderValue) =>
                  handleSliderChange('stt', 'beamSize', value)
                }
              />
            </div>
            <div className="grid gap-2">
              <Label>Temperature ({modelParams.stt.temperature})</Label>
              <Slider
                value={[modelParams.stt.temperature]}
                min={0}
                max={1}
                step={0.1}
                onValueChange={(value: SliderValue) =>
                  handleSliderChange('stt', 'temperature', value)
                }
              />
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="font-medium">Text-to-Speech</h3>
            <div className="grid gap-2">
              <Label>Speed ({modelParams.tts.speed}x)</Label>
              <Slider
                value={[modelParams.tts.speed]}
                min={0.5}
                max={2}
                step={0.1}
                onValueChange={(value: SliderValue) =>
                  handleSliderChange('tts', 'speed', value)
                }
              />
            </div>
            <div className="grid gap-2">
              <Label>Pitch ({modelParams.tts.pitch}x)</Label>
              <Slider
                value={[modelParams.tts.pitch]}
                min={0.5}
                max={2}
                step={0.1}
                onValueChange={(value: SliderValue) =>
                  handleSliderChange('tts', 'pitch', value)
                }
              />
            </div>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
