import { Button } from './ui'
import { Mic, Square, Loader2 } from 'lucide-react'

interface RecordingControlsProps {
  isRecording: boolean
  isProcessing: boolean
  isInitializing?: boolean
  onStartRecording: () => void
  onStopRecording: () => void
}

export function RecordingControls({
  isRecording,
  isProcessing,
  isInitializing = false,
  onStartRecording,
  onStopRecording
}: RecordingControlsProps) {
  return (
    <div className="sticky bottom-0 border-t bg-background p-4">
      <div className="flex justify-center">
        {isProcessing || isInitializing ? (
          <Button disabled className="w-12 h-12 rounded-full">
            <Loader2 className="w-6 h-6 animate-spin" />
          </Button>
        ) : isRecording ? (
          <Button
            onClick={onStopRecording}
            variant="destructive"
            className="w-12 h-12 rounded-full"
          >
            <Square className="w-6 h-6" />
          </Button>
        ) : (
          <Button
            onClick={onStartRecording}
            variant="default"
            className="w-12 h-12 rounded-full"
          >
            <Mic className="w-6 h-6" />
          </Button>
        )}
      </div>
    </div>
  )
}
