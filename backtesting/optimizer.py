import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import itertools
from concurrent.futures import ProcessPoolExecutor
import logging
from .backtest_engine import BacktestEngine

class StrategyOptimizer:
    def __init__(self, strategy_class, param_grid: Dict[str, List[Any]], metric: str = 'sharpe_ratio'):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.metric = metric
        self.logger = logging.getLogger(__name__)
        self.results = []

    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all possible parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _run_backtest(self, params: Dict, data: pd.DataFrame) -> Tuple[Dict, float]:
        """Run backtest with given parameters."""
        try:
            strategy = self.strategy_class(**params)
            engine = BacktestEngine(strategy)
            metrics = engine.run(data)
            return params, metrics[self.metric]
        except Exception as e:
            self.logger.error(f"Error running backtest with params {params}: {str(e)}")
            return params, float('-inf')

    def optimize(self, data: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        """Optimize strategy parameters using parallel processing."""
        param_combinations = self._generate_param_combinations()
        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Run backtests in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._run_backtest, params, data)
                for params in param_combinations
            ]
            
            # Collect results
            for future in futures:
                params, metric_value = future.result()
                self.results.append({
                    'params': params,
                    self.metric: metric_value
                })

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = pd.concat([
            results_df.drop(['params'], axis=1),
            pd.DataFrame(results_df['params'].tolist())
        ], axis=1)
        
        # Sort by metric value
        results_df = results_df.sort_values(by=self.metric, ascending=False)
        
        return results_df

    def get_best_params(self) -> Dict:
        """Get the best parameters found during optimization."""
        if not self.results:
            raise ValueError("No optimization results available")
        
        best_result = max(self.results, key=lambda x: x[self.metric])
        return best_result['params']

    def plot_optimization_results(self, save_path: str = None):
        """Plot optimization results."""
        if not self.results:
            raise ValueError("No optimization results available")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = pd.concat([
            results_df.drop(['params'], axis=1),
            pd.DataFrame(results_df['params'].tolist())
        ], axis=1)
        
        # Create pairplot
        g = sns.pairplot(
            results_df,
            vars=list(self.param_grid.keys()) + [self.metric],
            diag_kind='kde'
        )
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_parameter_importance(self) -> pd.DataFrame:
        """Calculate parameter importance using correlation analysis."""
        if not self.results:
            raise ValueError("No optimization results available")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = pd.concat([
            results_df.drop(['params'], axis=1),
            pd.DataFrame(results_df['params'].tolist())
        ], axis=1)
        
        # Calculate correlations
        correlations = {}
        for param in self.param_grid.keys():
            correlation = np.corrcoef(
                results_df[param],
                results_df[self.metric]
            )[0, 1]
            correlations[param] = abs(correlation)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'parameter': list(correlations.keys()),
            'importance': list(correlations.values())
        })
        
        return importance_df.sort_values('importance', ascending=False) 