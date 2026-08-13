import json
from typing import Any, Dict, List
import pandas as pd

class ValidationReport:
    def __init__(self, status: str, errors: List[str], warnings: List[str], infos: List[str], statistics: Dict[str, Any], df: pd.DataFrame = None):
        self.status = status
        self.errors = errors
        self.warnings = warnings
        self.infos = infos
        self.statistics = statistics
        self.df = df # Validated dataframe, which might have renamed columns
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.infos,
            "statistics": self.statistics
        }
        
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
        
    def __str__(self) -> str:
        report = []
        report.append("Smart Stock Data Validation Report")
        report.append("===================================")
        report.append(f"\nDataset Status: {self.status}\n")
        
        report.append(f"Rows: {self.statistics.get('rows', 0)}")
        report.append(f"Products: {self.statistics.get('products', 0)}")
        report.append(f"Stores: {self.statistics.get('stores', 0)}")
        
        date_range = self.statistics.get('date_range')
        if date_range:
            report.append(f"Date Range: {date_range[0]} -> {date_range[1]}\n")
        else:
            report.append("Date Range: Unknown\n")
            
        if self.errors:
            report.append("Errors:")
            for e in self.errors:
                report.append(f"  - {e}")
            report.append("")
            
        if self.warnings:
            report.append("Warnings:")
            for w in self.warnings:
                report.append(f"  - {w}")
            report.append("")
            
        if self.infos:
            report.append("Infos:")
            for i in self.infos:
                report.append(f"  - {i}")
            report.append("")
            
        return "\n".join(report)
