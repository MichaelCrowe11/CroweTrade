import { create } from 'zustand'

export interface MarketData {
  symbol: string
  price: number
  volume: number
  timestamp: number
  quantumState: 'superposition' | 'collapsed' | 'entangled'
}

export interface AgentMessage {
  agentId: string
  type: 'signal' | 'decision' | 'execution' | 'risk'
  content: any
  timestamp: number
  energy: number
}

export interface PortfolioUpdate {
  positions: Map<string, number>
  pnl: number
  drawdown: number
  var: number
  sharpe: number
}

interface WebSocketState {
  ws: WebSocket | null
  connected: boolean
  marketData: Map<string, MarketData>
  agentMessages: AgentMessage[]
  portfolio: PortfolioUpdate | null
  connect: (url: string) => void
  disconnect: () => void
  sendMessage: (message: any) => void
}

export const useWebSocket = create<WebSocketState>((set, get) => ({
  ws: null,
  connected: false,
  marketData: new Map(),
  agentMessages: [],
  portfolio: null,

  connect: (url: string) => {
    const ws = new WebSocket(url)

    ws.onopen = () => {
      console.log('WebSocket connected')
      set({ ws, connected: true })
      
      // Subscribe to channels
      ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['market', 'agents', 'portfolio', 'risk']
      }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        switch (data.type) {
          case 'market':
            set(state => {
              const newMarketData = new Map(state.marketData)
              newMarketData.set(data.symbol, {
                symbol: data.symbol,
                price: data.price,
                volume: data.volume,
                timestamp: data.timestamp,
                quantumState: data.quantumState || 'superposition'
              })
              return { marketData: newMarketData }
            })
            break
            
          case 'agent':
            set(state => ({
              agentMessages: [...state.agentMessages.slice(-99), {
                agentId: data.agentId,
                type: data.messageType,
                content: data.content,
                timestamp: data.timestamp,
                energy: data.energy || 50
              }]
            }))
            break
            
          case 'portfolio':
            set({
              portfolio: {
                positions: new Map(Object.entries(data.positions || {})),
                pnl: data.pnl,
                drawdown: data.drawdown,
                var: data.var,
                sharpe: data.sharpe
              }
            })
            break
            
          case 'quantum_event':
            // Handle special quantum events
            if (data.event === 'collapse') {
              console.log('Wave function collapsed:', data)
            } else if (data.event === 'entanglement') {
              console.log('Quantum entanglement detected:', data)
            }
            break
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      set({ ws: null, connected: false })
      
      // Reconnect after 5 seconds
      setTimeout(() => {
        if (!get().connected) {
          get().connect(url)
        }
      }, 5000)
    }

    set({ ws })
  },

  disconnect: () => {
    const { ws } = get()
    if (ws) {
      ws.close()
      set({ ws: null, connected: false })
    }
  },

  sendMessage: (message: any) => {
    const { ws, connected } = get()
    if (ws && connected) {
      ws.send(JSON.stringify(message))
    }
  }
}))

// Mock WebSocket server for development
export function createMockWebSocketServer() {
  if (typeof window === 'undefined') return

  // Simulate market data updates
  setInterval(() => {
    const store = useWebSocket.getState()
    if (!store.connected) return

    const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    const symbol = symbols[Math.floor(Math.random() * symbols.length)]
    
    const currentData = store.marketData.get(symbol)
    const basePrice = currentData?.price || 100 + Math.random() * 400
    
    const mockData = {
      type: 'market',
      symbol,
      price: basePrice + (Math.random() - 0.5) * 10,
      volume: Math.floor(Math.random() * 1000000),
      timestamp: Date.now(),
      quantumState: Math.random() > 0.7 ? 'collapsed' : 'superposition'
    }
    
    // Simulate receiving message
    store.marketData.set(symbol, mockData as any)
  }, 1000)

  // Simulate agent messages
  setInterval(() => {
    const store = useWebSocket.getState()
    if (!store.connected) return

    const agents = ['signal-1', 'portfolio-1', 'risk-1', 'execution-1']
    const types = ['signal', 'decision', 'execution', 'risk'] as const
    
    const mockMessage: AgentMessage = {
      agentId: agents[Math.floor(Math.random() * agents.length)],
      type: types[Math.floor(Math.random() * types.length)],
      content: { message: 'Processing quantum state transition' },
      timestamp: Date.now(),
      energy: 50 + Math.random() * 50
    }
    
    useWebSocket.setState(state => ({
      agentMessages: [...state.agentMessages.slice(-99), mockMessage]
    }))
  }, 3000)

  // Simulate portfolio updates
  setInterval(() => {
    const store = useWebSocket.getState()
    if (!store.connected) return

    const positions = new Map([
      ['AAPL', Math.random() * 100 - 50],
      ['GOOGL', Math.random() * 100 - 50],
      ['MSFT', Math.random() * 100 - 50],
    ])

    const mockPortfolio: PortfolioUpdate = {
      positions,
      pnl: (Math.random() - 0.5) * 10000,
      drawdown: Math.random() * 5,
      var: Math.random() * 1000,
      sharpe: Math.random() * 3
    }
    
    useWebSocket.setState({ portfolio: mockPortfolio })
  }, 5000)
}