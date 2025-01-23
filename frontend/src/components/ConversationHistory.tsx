import { Sheet, SheetContent, SheetHeader, SheetTitle, Button } from './ui'
import { History, Trash2 } from 'lucide-react'

interface ConversationHistoryProps {
  conversations: { id: number; started_at: string }[]
  showHistory: boolean
  onNewConversation: () => void
  onLoadConversation: (id: number) => void
  onDeleteConversation: (id: number) => void
  onShowHistoryChange: (show: boolean) => void
}

export function ConversationHistory({
  conversations,
  showHistory,
  onNewConversation,
  onLoadConversation,
  onDeleteConversation,
  onShowHistoryChange
}: ConversationHistoryProps) {
  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => onShowHistoryChange(true)}
      >
        <History className="h-5 w-5" />
      </Button>

      <Sheet open={showHistory} onOpenChange={onShowHistoryChange}>
        <SheetContent side="left">
          <SheetHeader>
            <SheetTitle>Conversation History</SheetTitle>
          </SheetHeader>
          <div className="mt-4 space-y-2">
            <Button
              onClick={() => {
                onNewConversation()
                onShowHistoryChange(false)
              }}
              className="w-full"
            >
              New Conversation
            </Button>
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className="flex items-center justify-between rounded-lg border p-2"
              >
                <Button
                  variant="ghost"
                  onClick={() => {
                    onLoadConversation(conv.id)
                    onShowHistoryChange(false)
                  }}
                  className="flex-1 justify-start"
                >
                  {new Date(conv.started_at).toLocaleString()}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => onDeleteConversation(conv.id)}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
