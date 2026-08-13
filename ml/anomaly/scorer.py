import pandas as pd
from typing import List, Dict
from .schema import AnomalyReport, AnomalyRecord, QualityScoreList

class DataQualityScorer:
    def evaluate(self, df_anomalies: pd.DataFrame) -> AnomalyReport:
        total_records = len(df_anomalies)
        if total_records == 0:
            return AnomalyReport(
                status="ERROR",
                total_records=0,
                anomalies=0,
                critical_anomalies=0,
                data_quality_score=0.0,
                product_scores=[],
                results=[]
            )
            
        anomalies_df = df_anomalies[df_anomalies['is_anomaly'] == True]
        total_anomalies = len(anomalies_df)
        
        critical_count = len(anomalies_df[anomalies_df['anomaly_severity'] == 'CRITICAL'])
        high_count = len(anomalies_df[anomalies_df['anomaly_severity'] == 'HIGH'])
        medium_count = len(anomalies_df[anomalies_df['anomaly_severity'] == 'MEDIUM'])
        low_count = len(anomalies_df[anomalies_df['anomaly_severity'] == 'LOW'])
        
        # Calculate Base Quality Score
        # Start at 100
        # Critical = -5 points each
        # High = -2 points each
        # Medium = -0.5 points each
        # Low = -0.1 points each
        penalty = (critical_count * 5) + (high_count * 2) + (medium_count * 0.5) + (low_count * 0.1)
        
        # Normalize penalty dynamically over the dataset size (e.g if 1000 records, 10 criticals shouldn't make score 0, but bad)
        normalized_penalty = (penalty / max(1, total_records)) * 100 # scales penalty linearly
        
        # Let's use a simpler bounded formula representing % of clean records weighted by severity
        weighted_errors = critical_count + (0.8 * high_count) + (0.3 * medium_count) + (0.05 * low_count)
        data_quality_score = max(0.0, 100.0 - ((weighted_errors / total_records) * 100.0))
        
        # Product Level Scores
        product_scores = []
        if 'product_id' in df_anomalies.columns:
            groups = df_anomalies.groupby(['store_id', 'product_id'])
            for (store, prod), group in groups:
                t_recs = len(group)
                c_err = len(group[group['anomaly_severity'] == 'CRITICAL'])
                h_err = len(group[group['anomaly_severity'] == 'HIGH'])
                m_err = len(group[group['anomaly_severity'] == 'MEDIUM'])
                l_err = len(group[group['anomaly_severity'] == 'LOW'])
                w_err = c_err + (0.8 * h_err) + (0.3 * m_err) + (0.05 * l_err)
                s_score = max(0.0, 100.0 - ((w_err / t_recs) * 100.0))
                product_scores.append(QualityScoreList(product_id=prod, store_id=store, score=round(s_score, 2)))
        
        # Extract individual results
        records = []
        for idx, row in anomalies_df.iterrows():
            records.append(AnomalyRecord(
                product_id=row.get('product_id', 'UNKNOWN'),
                store_id=row.get('store_id', 'STORE_01'),
                date=str(row['date'])[:10] if 'date' in row else 'UNKNOWN',
                anomaly=True,
                anomaly_type=row['anomaly_type'],
                severity=row['anomaly_severity'],
                score=row['anomaly_score'],
                reason=row['anomaly_reason'],
                original_value=float(row.get('sales', 0.0))
            ))
            
        status = "HEALTHY"
        if data_quality_score < 70:
            status = "CRITICAL"
        elif data_quality_score < 90 or critical_count > 0:
            status = "WARNING"
            
        return AnomalyReport(
            status=status,
            total_records=total_records,
            anomalies=total_anomalies,
            critical_anomalies=critical_count,
            data_quality_score=round(data_quality_score, 2),
            product_scores=product_scores,
            results=records
        )
