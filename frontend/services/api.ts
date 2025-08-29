const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface Signal {
  id: string
  timestamp: string
  symbol: string
  direction: 'long' | 'short' | 'neutral'
  confidence: number
  metadata: Record<string, any>
}

export interface TargetPosition {
  timestamp: string
  symbol: string
  target_weight: number
  current_weight: number
  reason: string
}

export interface RiskMetrics {
  portfolio_var: number
  max_drawdown: number
  current_drawdown: number
  sharpe_ratio: number
  sortino_ratio: number
  correlation_matrix: number[][]
}

export interface AgentStatus {
  agent_id: string
  name: string
  status: 'running' | 'stopped' | 'error'
  last_heartbeat: string
  metrics: Record<string, any>
}

class ApiService {
  private headers = {
    'Content-Type': 'application/json',
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        ...this.headers,
        ...options.headers,
      },
    })

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }

  // Signal Management
  async getSignals(limit = 100): Promise<Signal[]> {
    return this.request<Signal[]>(`/api/signals?limit=${limit}`)
  }

  async createSignal(signal: Omit<Signal, 'id'>): Promise<Signal> {
    return this.request<Signal>('/api/signals', {
      method: 'POST',
      body: JSON.stringify(signal),
    })
  }

  // Portfolio Management
  async getTargetPositions(): Promise<TargetPosition[]> {
    return this.request<TargetPosition[]>('/api/portfolio/targets')
  }

  async updateTargetPosition(position: TargetPosition): Promise<TargetPosition> {
    return this.request<TargetPosition>('/api/portfolio/targets', {
      method: 'PUT',
      body: JSON.stringify(position),
    })
  }

  async rebalancePortfolio(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(
      '/api/portfolio/rebalance',
      { method: 'POST' }
    )
  }

  // Risk Management
  async getRiskMetrics(): Promise<RiskMetrics> {
    return this.request<RiskMetrics>('/api/risk/metrics')
  }

  async updateRiskLimits(limits: {
    max_var: number
    max_drawdown: number
    position_limit: number
  }): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('/api/risk/limits', {
      method: 'PUT',
      body: JSON.stringify(limits),
    })
  }

  async triggerKillSwitch(): Promise<{ success: boolean; message: string }> {
    return this.request<{ success: boolean; message: string }>(
      '/api/risk/kill-switch',
      { method: 'POST' }
    )
  }

  // Agent Management
  async getAgentStatuses(): Promise<AgentStatus[]> {
    return this.request<AgentStatus[]>('/api/agents/status')
  }

  async startAgent(agentId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/api/agents/${agentId}/start`, {
      method: 'POST',
    })
  }

  async stopAgent(agentId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/api/agents/${agentId}/stop`, {
      method: 'POST',
    })
  }

  async restartAgent(agentId: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>(`/api/agents/${agentId}/restart`, {
      method: 'POST',
    })
  }

  // Quantum State Management
  async getQuantumState(): Promise<{
    coherence: number
    entanglements: Array<{ source: string; target: string; strength: number }>
    superposition_states: number
    observer_active: boolean
  }> {
    return this.request('/api/quantum/state')
  }

  async collapseWaveFunction(observerId: string): Promise<{
    collapsed_state: any
    measurement_outcome: string
  }> {
    return this.request('/api/quantum/collapse', {
      method: 'POST',
      body: JSON.stringify({ observer_id: observerId }),
    })
  }

  async entangleAgents(
    agent1: string,
    agent2: string,
    strength: number
  ): Promise<{ success: boolean }> {
    return this.request('/api/quantum/entangle', {
      method: 'POST',
      body: JSON.stringify({ agent1, agent2, strength }),
    })
  }

  // Market Data
  async getMarketSnapshot(): Promise<
    Array<{
      symbol: string
      price: number
      change: number
      volume: number
      bid: number
      ask: number
    }>
  > {
    return this.request('/api/market/snapshot')
  }

  async getHistoricalData(
    symbol: string,
    interval: '1m' | '5m' | '1h' | '1d',
    limit = 100
  ): Promise<
    Array<{
      timestamp: string
      open: number
      high: number
      low: number
      close: number
      volume: number
    }>
  > {
    return this.request(
      `/api/market/historical/${symbol}?interval=${interval}&limit=${limit}`
    )
  }

  // Execution
  async executeOrder(order: {
    symbol: string
    side: 'buy' | 'sell'
    quantity: number
    order_type: 'market' | 'limit'
    limit_price?: number
  }): Promise<{
    order_id: string
    status: 'pending' | 'filled' | 'rejected'
    filled_quantity: number
    average_price: number
  }> {
    return this.request('/api/execution/order', {
      method: 'POST',
      body: JSON.stringify(order),
    })
  }

  async getOrderStatus(orderId: string): Promise<{
    order_id: string
    status: string
    filled_quantity: number
    remaining_quantity: number
    average_price: number
    updates: Array<{ timestamp: string; status: string; message: string }>
  }> {
    return this.request(`/api/execution/order/${orderId}`)
  }

  // System Health
  async getSystemHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'critical'
    components: Array<{
      name: string
      status: 'up' | 'down'
      latency_ms: number
      error_rate: number
    }>
    uptime_seconds: number
    memory_usage_mb: number
    cpu_usage_percent: number
  }> {
    return this.request('/api/health')
  }

  // Backtesting
  async runBacktest(params: {
    start_date: string
    end_date: string
    initial_capital: number
    strategy_params: Record<string, any>
  }): Promise<{
    backtest_id: string
    status: 'running' | 'completed' | 'failed'
    progress: number
  }> {
    return this.request('/api/backtest/run', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  }

  async getBacktestResults(backtestId: string): Promise<{
    returns: number[]
    sharpe_ratio: number
    max_drawdown: number
    win_rate: number
    profit_factor: number
    trades: Array<{
      timestamp: string
      symbol: string
      side: string
      quantity: number
      price: number
      pnl: number
    }>
  }> {
    return this.request(`/api/backtest/${backtestId}/results`)
  }
}

export const apiService = new ApiService()

// Export quantum-specific API functions
export const quantumApi = {
  getQuantumState: () => apiService.getQuantumState(),
  collapseWaveFunction: (observerId: string) =>
    apiService.collapseWaveFunction(observerId),
  entangleAgents: (agent1: string, agent2: string, strength: number) =>
    apiService.entangleAgents(agent1, agent2, strength),
}

// Export trading-specific API functions
export const tradingApi = {
  getSignals: (limit?: number) => apiService.getSignals(limit),
  createSignal: (signal: Omit<Signal, 'id'>) => apiService.createSignal(signal),
  getTargetPositions: () => apiService.getTargetPositions(),
  executeOrder: (order: Parameters<typeof apiService.executeOrder>[0]) =>
    apiService.executeOrder(order),
}

// Export risk-specific API functions
export const riskApi = {
  getRiskMetrics: () => apiService.getRiskMetrics(),
  updateRiskLimits: (limits: Parameters<typeof apiService.updateRiskLimits>[0]) =>
    apiService.updateRiskLimits(limits),
  triggerKillSwitch: () => apiService.triggerKillSwitch(),
}