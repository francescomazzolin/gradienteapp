SCHEMA_REGISTRY = {
    "business_report_companies": {
        "name": "business_report_companies",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "description": "A list of company names within the group being analyzed.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["companies"],
            "additionalProperties": False
        }
    },

    
    "product_portfolio_impact": {
    "name": "product_portfolio_impact",
    "schema": {
        "type": "object",
        "properties": {
        "product_name": {
            "type": "string",
            "description": "The name of the product."
        },
        "impact_percentage": {
            "type": "number",
            "description": "The percentage value of the product's impact on overall sales."
        }
        },
        "required": [
        "product_name",
        "impact_percentage"
        ],
        
    },
    "strict": True,
    "additionalProperties": False
    },

    "market_analysis_request": {
    "name": "market_analysis_request",
    
    "schema": {
        "type": "object",
        "properties": {
        "market_sectors": {
            "type": "array",
            "description": "A list of markets or sectors for which market analysis is requested.",
            "items": {
            "type": "string",
            "description": "A specific market or sector."
            }
        }
        },
        "required": [
        "market_sectors"
        ],
        
    
    
        "strict": True,
        "additionalProperties": False
        }

    }




}