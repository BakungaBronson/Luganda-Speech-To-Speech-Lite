import { useState, useRef, useEffect } from 'react'
import { loadSettings, saveSettings } from './lib/storage'
import { Toaster } from './components/ui/toaster'
import { useToast } from './components/ui/use-toast'
import {
  SettingsPanel,
  ConversationHistory,
  ChatMessages,
  RecordingControls,
} from './components'
import { ModelParams, Message, ProviderType } from './types'

interface MediaRecorderError extends Error {
  name: string;
}

interface ExtendedMediaRecorderEvent extends Event {
  error: MediaRecorderError;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null)
  const [conversations, setConversations] = useState<{ id: number, started_at: string }[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const [modelParams, setModelParams] = useState<ModelParams>({
    stt: {
      beamSize: 5,
      temperature: 0.0,
    },
    tts: {
      speed: 1.0,
      pitch: 1.0,
    },
    api: {
      url: 'http://localhost:8000',
      providerType: 'openai',
      apiKey: '',
      baseUrl: '',
      modelName: ''
    }
  })

  useEffect(() => {
    const loadStoredSettings = async () => {
      const settings = await loadSettings();
      if (settings) {
        setModelParams(prev => ({
          ...prev,
          api: {
            ...prev.api,
            url: settings.apiUrl || prev.api.url,
            apiKey: settings.apiKey || '',
            providerType: (settings.providerType as ProviderType) || 'openai',
            baseUrl: settings.baseUrl || '',
            modelName: settings.modelName || ''
          },
          stt: {
            ...prev.stt,
            ...(settings.stt || {})
          },
          tts: {
            ...prev.tts,
            ...(settings.tts || {})
          }
        }));
      }
    };
    
    loadStoredSettings();
    loadConversations();
  }, [])

  useEffect(() => {
    saveSettings({
      apiUrl: modelParams.api.url,
      apiKey: modelParams.api.apiKey,
      providerType: modelParams.api.providerType,
      baseUrl: modelParams.api.baseUrl,
      modelName: modelParams.api.modelName,
      stt: modelParams.stt,
      tts: modelParams.tts
    });
  }, [modelParams])

  const showError = (message: string) => {
    toast({
      variant: "destructive",
      title: "Error",
      description: message,
    })
  }

  const loadConversations = async () => {
    try {
      const response = await fetch(`${modelParams.api.url}/api/v1/conversations`)
      if (!response.ok) {
        throw new Error('Failed to load conversations')
      }
      const data = await response.json()
      setConversations(data)
    } catch (error) {
      console.error('Error loading conversations:', error)
      showError('Failed to load conversations')
    }
  }

  const startNewConversation = async () => {
    try {
      const response = await fetch(`${modelParams.api.url}/api/v1/conversations`, {
        method: 'POST'
      })
      if (!response.ok) {
        throw new Error('Failed to start conversation')
      }
      const data = await response.json()
      setCurrentConversationId(data.conversation_id)
      setMessages([])
      await loadConversations()
    } catch (error) {
      console.error('Error starting conversation:', error)
      showError('Failed to start new conversation')
    }
  }

  const loadConversation = async (id: number) => {
    try {
      const response = await fetch(`${modelParams.api.url}/api/v1/conversations/${id}`)
      if (!response.ok) {
        throw new Error('Failed to load conversation')
      }
      const data = await response.json()
      setCurrentConversationId(id)
      
      // Create audio URLs for messages with audio paths
      const messagesWithAudio = data.messages.map((msg: Message) => {
        if (msg.audio_url) {
          return {
            ...msg,
            audio_url: `${modelParams.api.url}/audio/${msg.audio_url}`
          }
        }
        return msg
      })
      
      setMessages(messagesWithAudio)
    } catch (error) {
      console.error('Error loading conversation:', error)
      showError('Failed to load conversation')
    }
  }

  const deleteConversation = async (id: number) => {
    try {
      const response = await fetch(`${modelParams.api.url}/api/v1/conversations/${id}`, {
        method: 'DELETE'
      })
      if (!response.ok) {
        throw new Error('Failed to delete conversation')
      }
      if (currentConversationId === id) {
        setCurrentConversationId(null)
        setMessages([])
      }
      await loadConversations()
    } catch (error) {
      console.error('Error deleting conversation:', error)
      showError('Failed to delete conversation')
    }
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
    if (!modelParams.api.apiKey) {
      showError('Please enter your API key in settings')
      return
    }

    if (isRecording || mediaRecorderRef.current?.state === 'recording') {
      console.warn('Already recording')
      return
    }

    if (!currentConversationId) {
      await startNewConversation()
    }

    try {
      cleanupRecording()

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      
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
        showError('Error recording audio: ' + recorderEvent.error.message)
      }

      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
          await processAudio(blob)
        } catch (error) {
          console.error('Error processing recorded audio:', error)
          showError('Error processing audio: ' + (error instanceof Error ? error.message : 'Unknown error'))
        } finally {
          cleanupRecording()
        }
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      showError('Error accessing microphone. Please ensure microphone permissions are granted.')
      cleanupRecording()
    }
  }

  const processAudio = async (inputBlob: Blob) => {
    if (!currentConversationId) {
      console.error('No active conversation')
      return
    }

    setIsProcessing(true)
    try {
      const formData = new FormData()
      formData.append('file', inputBlob)
      formData.append('conversation_id', currentConversationId.toString())

      // Step 1: Transcribe audio
      const transcribeResponse = await fetch(`${modelParams.api.url}/api/v1/transcribe`, {
        method: 'POST',
        body: formData,
      })
      
      if (!transcribeResponse.ok) {
        const errorData = await transcribeResponse.json().catch(() => ({}))
        console.error('Transcription error details:', errorData)
        throw new Error(errorData.detail || 'Failed to transcribe audio')
      }
      
      const transcribeData = await transcribeResponse.json()

      // Add user message
      const userMessage: Message = {
        role: 'user',
        content: transcribeData.text,
      }
      setMessages(prev => [...prev, userMessage])

      // Step 2: Process with chat API
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'X-Provider-Type': modelParams.api.providerType,
        'X-API-Key': modelParams.api.apiKey,
      }

      if (modelParams.api.providerType === 'custom') {
        headers['X-Base-URL'] = modelParams.api.baseUrl || ''
        headers['X-Model-Name'] = modelParams.api.modelName || ''
      }
      
      const chatResponse = await fetch(`${modelParams.api.url}/api/v1/chat`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          text: transcribeData.text,
          conversation_id: currentConversationId
        }),
      })

      if (!chatResponse.ok) {
        const errorData = await chatResponse.json().catch(() => ({}))
        console.error('Chat API error details:', errorData)
        throw new Error(errorData.detail || 'Failed to process with chat API')
      }

      const chatData = await chatResponse.json()
      const responseText = chatData.text

      if (chatData.conversation) {
        const updatedMessages = chatData.conversation.messages.map((msg: Message) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          audio_url: msg.audio_url,
          created_at: msg.created_at
        }))
        setMessages(updatedMessages)
        return
      }

      // Step 3: Synthesize speech
      const synthesizeResponse = await fetch(`${modelParams.api.url}/api/v1/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/wav',
        },
        body: JSON.stringify({
          text: responseText,
          conversation_id: currentConversationId,
          speed: modelParams.tts.speed,
          pitch: modelParams.tts.pitch
        }),
      })

      if (!synthesizeResponse.ok) {
        const errorData = await synthesizeResponse.json().catch(() => ({}))
        console.error('Synthesis error details:', errorData)
        throw new Error(errorData.detail || 'Failed to synthesize speech')
      }

      const responseBlob = await synthesizeResponse.blob()
      const audioUrl = URL.createObjectURL(responseBlob)

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: responseText,
        audio_url: audioUrl,
      }
      setMessages(prev => [...prev, assistantMessage])
      
      // Refresh conversation to ensure sync with DB
      await loadConversation(currentConversationId)

      // Scroll to bottom
      scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
    } catch (error) {
      console.error('Error processing audio:', error)
      showError(error instanceof Error ? error.message : 'Error processing audio. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <>
      <Toaster />
      <div className="flex h-screen flex-col">
        <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b bg-background px-4">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">Luganda Speech-to-Speech</h1>
            <ConversationHistory
              conversations={conversations}
              showHistory={showHistory}
              onNewConversation={startNewConversation}
              onLoadConversation={loadConversation}
              onDeleteConversation={deleteConversation}
              onShowHistoryChange={setShowHistory}
            />
          </div>
          <SettingsPanel
            modelParams={modelParams}
            onSettingsChange={setModelParams}
          />
        </header>

        <ChatMessages
          messages={messages}
          scrollRef={scrollRef}
          onError={showError}
        />

        <RecordingControls
          isRecording={isRecording}
          isProcessing={isProcessing}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
        />
      </div>
    </>
  )
}

export default App
