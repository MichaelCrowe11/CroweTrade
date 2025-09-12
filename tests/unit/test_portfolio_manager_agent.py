from crowetrade.core.agent import AgentConfig
from crowetrade.live.portfolio_manager import PortfolioManagerAgent


def test_expected_returns_with_partial_predictions():
    config = AgentConfig(agent_id="PM-1", policy_id="policy-test")
    agent = PortfolioManagerAgent(config, default_expected_return=0.08)

    symbols = ["AAPL", "GOOGL"]
    predictions = {"AAPL": 0.10}  # Only one prediction provided

    returns = agent._get_expected_returns(symbols, predictions)

    assert returns["AAPL"] == 0.10
    assert returns["GOOGL"] == 0.08  # Fallback default
