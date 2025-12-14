"""
Tests pour la validation Pydantic de l'agent Analyst.

Vérifie que:
- La validation accepte les structures valides
- La validation rejette les structures invalides
- Les messages d'erreur sont explicites
"""

import pytest
from pydantic import ValidationError

from agents.analyst import (
    AnalysisResponse,
    KeyMetricsAssessment,
    MetricAssessment,
)


class TestMetricAssessment:
    """Tests pour le modèle MetricAssessment."""
    
    def test_valid_metric_assessment(self):
        """Test avec données valides."""
        metric = MetricAssessment(
            value=1.5,
            assessment="Good Sharpe ratio indicating positive risk-adjusted returns"
        )
        
        assert metric.value == 1.5
        assert len(metric.assessment) > 0
    
    def test_missing_value(self):
        """Test avec valeur manquante."""
        with pytest.raises(ValidationError) as exc_info:
            MetricAssessment(assessment="test")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("value",)
        assert errors[0]["type"] == "missing"  # Pydantic v2
    
    def test_empty_assessment(self):
        """Test avec assessment vide."""
        with pytest.raises(ValidationError) as exc_info:
            MetricAssessment(value=1.5, assessment="")
        
        errors = exc_info.value.errors()
        # Pydantic v2 : type string_too_short
        assert any("string" in err["type"] and "short" in err["type"] for err in errors)
    
    def test_invalid_value_type(self):
        """Test avec type invalide pour value."""
        with pytest.raises(ValidationError) as exc_info:
            MetricAssessment(value="not_a_number", assessment="test")
        
        errors = exc_info.value.errors()
        assert any("float" in str(err).lower() for err in errors)


class TestKeyMetricsAssessment:
    """Tests pour le modèle KeyMetricsAssessment."""
    
    def test_valid_key_metrics(self):
        """Test avec toutes les métriques valides."""
        metrics = KeyMetricsAssessment(
            sharpe_ratio=MetricAssessment(value=1.5, assessment="Good Sharpe"),
            max_drawdown=MetricAssessment(value=-0.15, assessment="Acceptable DD"),
            win_rate=MetricAssessment(value=0.55, assessment="Above 50%"),
            profit_factor=MetricAssessment(value=1.8, assessment="Solid PF")
        )
        
        assert metrics.sharpe_ratio.value == 1.5
        assert metrics.max_drawdown.value == -0.15
        assert metrics.win_rate.value == 0.55
        assert metrics.profit_factor.value == 1.8
    
    def test_missing_sharpe_ratio(self):
        """Test avec sharpe_ratio manquant."""
        with pytest.raises(ValidationError) as exc_info:
            KeyMetricsAssessment(
                max_drawdown=MetricAssessment(value=-0.15, assessment="test"),
                win_rate=MetricAssessment(value=0.55, assessment="test"),
                profit_factor=MetricAssessment(value=1.8, assessment="test")
            )
        
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("sharpe_ratio",) for err in errors)


class TestAnalysisResponse:
    """Tests pour le modèle AnalysisResponse complet."""
    
    @pytest.fixture
    def valid_analysis_data(self):
        """Données d'analyse valides."""
        return {
            "summary": "This strategy shows strong performance with consistent returns.",
            "performance_rating": "GOOD",
            "risk_rating": "MODERATE",
            "overfitting_risk": "LOW",
            "strengths": [
                "High Sharpe ratio",
                "Consistent win rate",
                "Good risk management"
            ],
            "weaknesses": [
                "Moderate drawdown",
                "Limited sample size"
            ],
            "concerns": [
                "Market regime dependency"
            ],
            "key_metrics_assessment": {
                "sharpe_ratio": {
                    "value": 1.52,
                    "assessment": "Strong risk-adjusted returns"
                },
                "max_drawdown": {
                    "value": -0.15,
                    "assessment": "Acceptable drawdown level"
                },
                "win_rate": {
                    "value": 0.57,
                    "assessment": "Above 50%, indicating edge"
                },
                "profit_factor": {
                    "value": 1.85,
                    "assessment": "Solid profit factor"
                }
            },
            "recommendations": [
                "Continue optimization with current parameters",
                "Test on longer timeframe",
                "Validate with out-of-sample data"
            ],
            "proceed_to_optimization": True,
            "reasoning": "The strategy demonstrates solid fundamentals and justifies further optimization."
        }
    
    def test_valid_analysis_complete(self, valid_analysis_data):
        """Test avec analyse complète valide."""
        analysis = AnalysisResponse.parse_obj(valid_analysis_data)
        
        assert analysis.summary == valid_analysis_data["summary"]
        assert analysis.performance_rating == "GOOD"
        assert analysis.risk_rating == "MODERATE"
        assert analysis.overfitting_risk == "LOW"
        assert len(analysis.strengths) == 3
        assert len(analysis.weaknesses) == 2
        assert len(analysis.concerns) == 1
        assert len(analysis.recommendations) == 3
        assert analysis.proceed_to_optimization is True
        assert analysis.key_metrics_assessment.sharpe_ratio.value == 1.52
    
    def test_missing_summary(self, valid_analysis_data):
        """Test avec summary manquant."""
        del valid_analysis_data["summary"]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("summary",) for err in errors)
    
    def test_invalid_performance_rating(self, valid_analysis_data):
        """Test avec performance_rating invalide."""
        valid_analysis_data["performance_rating"] = "INVALID_RATING"
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any("performance_rating" in str(err["loc"]) for err in errors)
        # Pydantic v2 : type string_pattern_mismatch
        assert any("pattern" in err["type"] for err in errors)
    
    def test_invalid_risk_rating(self, valid_analysis_data):
        """Test avec risk_rating invalide."""
        valid_analysis_data["risk_rating"] = "SUPER_HIGH"
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any("risk_rating" in str(err["loc"]) for err in errors)
    
    def test_invalid_overfitting_risk(self, valid_analysis_data):
        """Test avec overfitting_risk invalide."""
        valid_analysis_data["overfitting_risk"] = "MAYBE"
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any("overfitting_risk" in str(err["loc"]) for err in errors)
    
    def test_empty_strengths_list(self, valid_analysis_data):
        """Test avec liste de strengths vide (acceptable)."""
        valid_analysis_data["strengths"] = []
        
        # Devrait passer car default_factory=list
        analysis = AnalysisResponse.parse_obj(valid_analysis_data)
        assert analysis.strengths == []
    
    def test_empty_string_in_strengths(self, valid_analysis_data):
        """Test avec string vide dans strengths (invalide)."""
        valid_analysis_data["strengths"] = ["Good metric", "", "Another strength"]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        # Le validateur custom doit rejeter les strings vides
        assert len(errors) > 0
    
    def test_whitespace_string_in_weaknesses(self, valid_analysis_data):
        """Test avec string whitespace dans weaknesses (invalide)."""
        valid_analysis_data["weaknesses"] = ["Valid weakness", "   ", "Another weakness"]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert len(errors) > 0
    
    def test_missing_key_metrics_assessment(self, valid_analysis_data):
        """Test avec key_metrics_assessment manquant."""
        del valid_analysis_data["key_metrics_assessment"]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("key_metrics_assessment",) for err in errors)
    
    def test_incomplete_key_metrics_assessment(self, valid_analysis_data):
        """Test avec key_metrics_assessment incomplet."""
        # Supprimer profit_factor
        del valid_analysis_data["key_metrics_assessment"]["profit_factor"]
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any("profit_factor" in str(err["loc"]) for err in errors)
    
    def test_invalid_proceed_to_optimization_type(self, valid_analysis_data):
        """Test avec proceed_to_optimization de type invalide (non convertible)."""
        valid_analysis_data["proceed_to_optimization"] = "maybe"  # Pas convertible en bool
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any("proceed_to_optimization" in str(err["loc"]) for err in errors)
    
    def test_short_summary(self, valid_analysis_data):
        """Test avec summary trop court."""
        valid_analysis_data["summary"] = "Short"  # < 10 caractères
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("summary",) for err in errors)
        # Pydantic v2 : type string_too_short
        assert any("string" in err["type"] and "short" in err["type"] for err in errors)
    
    def test_short_reasoning(self, valid_analysis_data):
        """Test avec reasoning trop court."""
        valid_analysis_data["reasoning"] = "Ok"  # < 10 caractères
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse.parse_obj(valid_analysis_data)
        
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("reasoning",) for err in errors)
    
    def test_all_valid_performance_ratings(self, valid_analysis_data):
        """Test avec tous les performance_rating valides."""
        valid_ratings = ["EXCELLENT", "GOOD", "FAIR", "POOR", "CRITICAL"]
        
        for rating in valid_ratings:
            valid_analysis_data["performance_rating"] = rating
            analysis = AnalysisResponse.parse_obj(valid_analysis_data)
            assert analysis.performance_rating == rating
    
    def test_all_valid_risk_ratings(self, valid_analysis_data):
        """Test avec tous les risk_rating valides."""
        valid_ratings = ["LOW", "MODERATE", "HIGH", "EXTREME"]
        
        for rating in valid_ratings:
            valid_analysis_data["risk_rating"] = rating
            analysis = AnalysisResponse.parse_obj(valid_analysis_data)
            assert analysis.risk_rating == rating
    
    def test_all_valid_overfitting_risks(self, valid_analysis_data):
        """Test avec tous les overfitting_risk valides."""
        valid_risks = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        
        for risk in valid_risks:
            valid_analysis_data["overfitting_risk"] = risk
            analysis = AnalysisResponse.parse_obj(valid_analysis_data)
            assert analysis.overfitting_risk == risk


class TestAnalystAgentValidation:
    """Tests d'intégration pour _validate_analysis de AnalystAgent."""
    
    @pytest.fixture
    def analyst_agent(self):
        """Créer un agent analyst pour les tests."""
        from agents.analyst import AnalystAgent
        from agents.llm_client import LLMConfig, LLMProvider
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="test-model",
            ollama_host="http://localhost:11434"
        )
        return AnalystAgent(config)
    
    @pytest.fixture
    def valid_analysis_dict(self):
        """Dictionnaire d'analyse valide."""
        return {
            "summary": "Strong performance with good risk management and consistent returns.",
            "performance_rating": "GOOD",
            "risk_rating": "MODERATE",
            "overfitting_risk": "LOW",
            "strengths": ["High Sharpe", "Good win rate"],
            "weaknesses": ["Moderate drawdown"],
            "concerns": ["Market dependency"],
            "key_metrics_assessment": {
                "sharpe_ratio": {"value": 1.5, "assessment": "Strong"},
                "max_drawdown": {"value": -0.15, "assessment": "Acceptable"},
                "win_rate": {"value": 0.55, "assessment": "Good"},
                "profit_factor": {"value": 1.8, "assessment": "Solid"}
            },
            "recommendations": ["Continue optimization"],
            "proceed_to_optimization": True,
            "reasoning": "Strategy shows solid fundamentals warranting further optimization."
        }
    
    def test_validate_analysis_success(self, analyst_agent, valid_analysis_dict):
        """Test validation réussie."""
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert errors == []  # Aucune erreur
    
    def test_validate_analysis_missing_field(self, analyst_agent, valid_analysis_dict):
        """Test avec champ manquant."""
        del valid_analysis_dict["summary"]
        
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert len(errors) > 0
        assert any("summary" in err.lower() for err in errors)
    
    def test_validate_analysis_invalid_enum(self, analyst_agent, valid_analysis_dict):
        """Test avec valeur enum invalide."""
        valid_analysis_dict["performance_rating"] = "INVALID"
        
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert len(errors) > 0
        assert any("performance_rating" in err for err in errors)
    
    def test_validate_analysis_invalid_type(self, analyst_agent, valid_analysis_dict):
        """Test avec type invalide."""
        valid_analysis_dict["proceed_to_optimization"] = "maybe"  # String au lieu de bool
        
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert len(errors) > 0
        assert any("proceed_to_optimization" in err for err in errors)
    
    def test_validate_analysis_incomplete_metrics(self, analyst_agent, valid_analysis_dict):
        """Test avec métriques incomplètes."""
        del valid_analysis_dict["key_metrics_assessment"]["sharpe_ratio"]
        
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert len(errors) > 0
        assert any("sharpe_ratio" in err for err in errors)
    
    def test_validate_analysis_empty_string_in_list(self, analyst_agent, valid_analysis_dict):
        """Test avec string vide dans une liste."""
        valid_analysis_dict["strengths"] = ["Valid", "", "Another"]
        
        errors = analyst_agent._validate_analysis(valid_analysis_dict)
        
        assert len(errors) > 0
    
    def test_validate_analysis_exception_handling(self, analyst_agent):
        """Test gestion d'exception (dict None)."""
        # Passer None au lieu d'un dict provoque ValidationError
        errors = analyst_agent._validate_analysis(None)
        
        assert len(errors) > 0
        # Pydantic génère un message avec "input" ou "none" ou field name
        assert any(
            any(keyword in err.lower() for keyword in ["input", "none", "summary", "champ"])
            for err in errors
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
