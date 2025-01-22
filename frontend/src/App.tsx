import { useState, useRef } from 'react'
import { Mic, Square, Loader2, Settings, Volume2 } from 'lucide-react'
import {
  Button,
  Slider,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  Label,
  ScrollArea
} from './components/ui'

interface MediaRecorderError extends Error {
  name: string;
}

interface ExtendedMediaRecorderEvent extends Event {
  error: MediaRecorderError;
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  audioUrl?: string
}

interface ModelParams {
  stt: {
    beamSize: number
    temperature: number
  }
  tts: {
    speed: number
    pitch: number
  }
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)

  const [modelParams, setModelParams] = useState<ModelParams>({
    stt: {
      beamSize: 5,
      temperature: 0.0,
    },
    tts: {
      speed: 1.0,
      pitch: 1.0,
    },
  })

  type SliderValue = [number]

  const handleSliderChange = (
    category: 'stt' | 'tts',
    param: 'beamSize' | 'temperature' | 'speed' | 'pitch',
    value: SliderValue
  ) => {
    setModelParams(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [param]: value[0]
      }
    }))
  }

  const cleanupRecording = () => {
    if (mediaRecorderRef.current?.stream) {
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
    }
    mediaRecorderRef.current = null
    audioChunksRef.current = []
    setIsRecording(false)
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }

  const startRecording = async () => {
    if (isRecording || mediaRecorderRef.current?.state === 'recording') {
      console.warn('Already recording')
      return
    }

    try {
      cleanupRecording()

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
      // Use WebM format for best compatibility with ffmpeg
      const options = {
        mimeType: 'audio/webm',
        audioBitsPerSecond: 16000
      }

      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        throw new Error('WebM audio recording is not supported in this browser')
      }
      
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onerror = (event: Event) => {
        const recorderEvent = event as ExtendedMediaRecorderEvent;
        console.error('MediaRecorder error:', recorderEvent.error)
        stopRecording()
        alert('Error recording audio: ' + recorderEvent.error.message)
      }

      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
          await processAudio(blob)
        } catch (error) {
          console.error('Error processing recorded audio:', error)
          alert('Error processing audio: ' + (error instanceof Error ? error.message : 'Unknown error'))
        } finally {
          cleanupRecording()
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Error accessing microphone. Please ensure microphone permissions are granted.')
      cleanupRecording()
    }
  }

  const processAudio = async (inputBlob: Blob) => {
    setIsProcessing(true)
    try {
      const formData = new FormData()
      formData.append('file', inputBlob)

      // Step 1: Transcribe audio
      const transcribeResponse = await fetch('http://localhost:8000/api/v1/transcribe', {
        method: 'POST',
        body: formData,
      })
      
      if (!transcribeResponse.ok) {
        throw new Error('Failed to transcribe audio')
      }
      
      const transcribeData = await transcribeResponse.json()

      // Add user message
      const userMessage: Message = {
        role: 'user',
        content: transcribeData.text,
      }
      setMessages(prev => [...prev, userMessage])

      // Step 2: Synthesize speech
      console.log('Sending text for synthesis:', transcribeData.text)
      const synthesizeResponse = await fetch('http://localhost:8000/api/v1/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        body: JSON.stringify({
          text: transcribeData.text
        }),
      })

      if (!synthesizeResponse.ok) {
        const errorData = await synthesizeResponse.json().catch(() => ({}))
        console.error('Synthesis error details:', errorData)
        throw new Error(errorData.detail || 'Failed to synthesize speech')
      }

      const responseBlob = await synthesizeResponse.blob()
      const audioUrl = URL.createObjectURL(responseBlob)

      // Add assistant message with audio
      const assistantMessage: Message = {
        role: 'assistant',
        content: transcribeData.text,
        audioUrl,
      }
      setMessages(prev => [...prev, assistantMessage])

      // Scroll to bottom
      scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
    } catch (error) {
      console.error('Error processing audio:', error)
      alert(error instanceof Error ? error.message : 'Error processing audio. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const playAudio = (audioUrl: string) => {
    const audio = new Audio(audioUrl)
    audio.play()
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b bg-background px-4">
        <h1 className="text-lg font-semibold">Luganda Speech-to-Speech</h1>
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
      </header>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4 max-w-2xl mx-auto">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`rounded-lg px-4 py-2 max-w-[80%] ${
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted'
                }`}
              >
                <p>{message.content}</p>
                {message.audioUrl && (
                  <Button
                    variant="secondary"
                    size="sm"
                    className="mt-2"
                    onClick={() => playAudio(message.audioUrl!)}
                  >
                    <Volume2 className="w-4 h-4 mr-2" />
                    Play Audio
                  </Button>
                )}
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      <div className="sticky bottom-0 border-t bg-background p-4">
        <div className="flex justify-center">
          {isProcessing ? (
            <Button disabled className="w-12 h-12 rounded-full">
              <Loader2 className="w-6 h-6 animate-spin" />
            </Button>
          ) : isRecording ? (
            <Button
              onClick={stopRecording}
              variant="destructive"
              className="w-12 h-12 rounded-full"
            >
              <Square className="w-6 h-6" />
            </Button>
          ) : (
            <Button
              onClick={startRecording}
              variant="default"
              className="w-12 h-12 rounded-full"
            >
              <Mic className="w-6 h-6" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
