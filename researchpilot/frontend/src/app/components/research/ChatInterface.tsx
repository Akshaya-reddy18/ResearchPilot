import { Send, Sparkles, File, User, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { chat, getCollectionStats, type ChatResponse } from '../../services/api';

interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sourceChunks?: number;
  model?: string;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      role: 'assistant',
      content: 'Hello! I\'m your AI research assistant. I can help you analyze papers, find insights, and answer questions about your research topics. What would you like to explore today?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [importedPapers, setImportedPapers] = useState<string[]>([]);
  const [documentCount, setDocumentCount] = useState<number>(0);

  // Load document stats on mount
  useEffect(() => {
    loadDocumentStats();
  }, []);

  const loadDocumentStats = async () => {
    try {
      const stats = await getCollectionStats();
      setDocumentCount(stats.data.document_count);
      
      // Load paper names from metadata if available
      // For now, we'll just show the count
    } catch (error) {
      console.error('Failed to load document stats:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: messages.length + 1,
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      console.log('Sending chat query:', input);
      const response: ChatResponse = await chat({
        query: input,
        use_context: true,
        top_k: 5,
      });
      console.log('Chat response:', response);

      const assistantMessage: Message = {
        id: messages.length + 2,
        role: 'assistant',
        content: response.analysis || 'I apologize, but I couldn\'t generate a response. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        sourceChunks: response.source_chunks_used,
        model: response.model,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      
      // Reload stats to update document count
      await loadDocumentStats();
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: messages.length + 2,
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response. Please check your connection and try again.'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex-1 flex overflow-hidden bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d]">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-indigo-900/30 bg-white/5 backdrop-blur-xl">
          <h1 className="text-2xl text-white flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            AI Research Assistant
          </h1>
          <p className="text-gray-400 text-sm mt-1 ml-13">Ask questions about your research papers</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((message) => (
            <div key={message.id} className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : ''}`}>
              {message.role === 'assistant' && (
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
              )}

              <div className={`max-w-3xl ${message.role === 'user' ? 'order-first' : ''}`}>
                <div
                  className={`p-4 rounded-2xl ${
                    message.role === 'user'
                      ? 'bg-indigo-500/20 border border-indigo-500/30 ml-auto'
                      : 'bg-white/5 border border-white/10'
                  }`}
                >
                  <div className="text-white whitespace-pre-line leading-relaxed">
                    {message.content.split('\n').map((line, i) => {
                      if (line.startsWith('**') && line.endsWith('**')) {
                        return (
                          <h3 key={i} className="font-semibold text-indigo-300 mt-3 mb-2 first:mt-0">
                            {line.replace(/\*\*/g, '')}
                          </h3>
                        );
                      }
                      if (line.startsWith('- ')) {
                        return (
                          <div key={i} className="flex gap-2 ml-4 mb-1">
                            <span className="text-indigo-400">•</span>
                            <span className="text-gray-300">{line.substring(2)}</span>
                          </div>
                        );
                      }
                      return line && <p key={i} className="text-gray-300 mb-2">{line}</p>;
                    })}
                  </div>
                </div>
                <div className="flex items-center justify-between mt-2 ml-2">
                  <p className="text-xs text-gray-500">{message.timestamp}</p>
                  {message.role === 'assistant' && message.sourceChunks !== undefined && (
                    <p className="text-xs text-gray-500">
                      {message.sourceChunks > 0 
                        ? `Used ${message.sourceChunks} source${message.sourceChunks !== 1 ? 's' : ''}${message.model ? ` • ${message.model}` : ''}`
                        : 'No sources used'}
                    </p>
                  )}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="w-10 h-10 rounded-lg bg-gray-700 flex items-center justify-center shrink-0">
                  <User className="w-5 h-5 text-gray-300" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="p-6 border-t border-indigo-900/30 bg-white/5 backdrop-blur-xl">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your research..."
              className="flex-1 px-4 py-3 bg-white/5 border border-indigo-900/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500/50 focus:bg-white/10 transition-all"
            />
            <button
              onClick={handleSend}
              disabled={isLoading}
              className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-all shadow-lg shadow-indigo-500/30 flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </div>

      {/* Sidebar - Imported Papers */}
      <div className="w-80 border-l border-indigo-900/30 bg-white/5 backdrop-blur-xl p-6">
        <h3 className="text-white mb-4 flex items-center gap-2">
          <File className="w-5 h-5 text-indigo-400" />
          Imported Papers
        </h3>
        <p className="text-xs text-gray-400 mb-4">
          {documentCount > 0 
            ? `${documentCount} document chunk${documentCount !== 1 ? 's' : ''} indexed`
            : 'No documents indexed yet'}
        </p>
        <div className="space-y-3">
          {importedPapers.length > 0 ? (
            importedPapers.map((paper, index) => (
              <div
                key={index}
                className="p-3 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all cursor-pointer"
              >
                <div className="flex items-start gap-2">
                  <File className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                  <p className="text-sm text-gray-300 leading-tight">{paper}</p>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 italic">
              Upload PDFs to see them here
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
