import { Message } from '../types'
import { ScrollArea, Button } from './ui'
import { Volume2 } from 'lucide-react'
import { RefObject } from 'react'

interface ChatMessagesProps {
  messages: Message[]
  scrollRef: RefObject<HTMLDivElement>
  onError: (message: string) => void
}

export function ChatMessages({ messages, scrollRef, onError }: ChatMessagesProps) {
  const handlePlayAudio = async (audioUrl: string) => {
    try {
      const audio = new Audio(audioUrl)
      await audio.play()
    } catch (error) {
      console.error('Error playing audio:', error)
      onError('Error playing audio. Please try again.')
    }
  }

  return (
    <ScrollArea className="flex-1 p-4">
      <div className="space-y-4 max-w-2xl mx-auto">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-[50vh] text-center space-y-4">
            <h2 className="text-2xl font-semibold">Welcome to Luganda Speech-to-Speech</h2>
            <p className="text-muted-foreground max-w-md">
              Click the microphone button below to start speaking in Luganda. 
              Your speech will be transcribed and you'll receive a response in Luganda with audio playback.
            </p>
            <p className="text-sm text-muted-foreground">
              Make sure your microphone is connected and permissions are granted.
            </p>
          </div>
        ) : messages.map((message, index) => (
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
              {message.audio_url && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="mt-2"
                  onClick={() => handlePlayAudio(message.audio_url!)}
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
  )
}
