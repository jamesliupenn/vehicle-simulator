"""
DIMO Vehicle Telemetry Data Generator using NeMo Data Designer Client

This script uses the NVIDIA NeMo Data Designer API to generate synthetic
vehicle telemetry data based on DIMO's data schema.

Prerequisites:
- NeMo Microservice running at http://localhost:8080
- nemo_microservices package installed
"""

from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    InferenceParameters,
    LLMTextColumnConfig,
    ModelConfig,
    NeMoDataDesignerClient,
    PersonSamplerParams,
    SamplerColumnConfig,
    SamplerType,
    SubcategorySamplerParams,
    UniformSamplerParams,
)
import json


# ============================================================
# STEP 1: Initialize NeMo Service Connection
# ============================================================

def initialize_nemo_client():
    """
    Initialize the NeMo Data Designer Client
    
    This connects to your locally running NeMo Microservice at localhost:8080

    To run this microservice, follow the guide here: https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/docker-compose.html#ndd-docker-compose
    """
    print("="*60)
    print("STEP 1: Initializing NeMo Data Designer Client")
    print("="*60)
    
    NEMO_MICROSERVICES_BASE_URL = "http://localhost:8080"
    data_designer_client = NeMoDataDesignerClient(base_url=NEMO_MICROSERVICES_BASE_URL)
    
    print(f"✓ Connected to: {NEMO_MICROSERVICES_BASE_URL}")
    print("✓ Client initialized successfully!\n")
    
    return data_designer_client


# ============================================================
# STEP 2: Configure the Model
# ============================================================

def setup_model_configuration():
    """
    Configure the LLM model for data generation
    
    This sets up which model to use and its parameters like temperature,
    max tokens, etc. These settings affect how creative or deterministic
    the generated data will be.
    """
    print("="*60)
    print("STEP 2: Setting up Model Configuration")
    print("="*60)
    
    # Model provider - set in microservice deployment configuration
    MODEL_PROVIDER = "nvidiabuild"
    
    # Model ID from build.nvidia.com
    MODEL_ID = "nvidia/nemotron-3-nano-30b-a3b"
    
    # Descriptive alias for our use case
    MODEL_ALIAS = "nemotron-nano-v3"
    
    # System prompt - disables reasoning for this model
    SYSTEM_PROMPT = "/no_think"
    
    print(f"Model Provider: {MODEL_PROVIDER}")
    print(f"Model ID: {MODEL_ID}")
    print(f"Model Alias: {MODEL_ALIAS}")
    
    # Create model configuration
    model_configs = [
        ModelConfig(
            alias=MODEL_ALIAS,
            model=MODEL_ID,
            provider=MODEL_PROVIDER,
            inference_parameters=InferenceParameters(
                temperature=0.25,      # Controls randomness (0=deterministic, 1=creative)
                top_p=1.0,           # Nucleus sampling parameter
                max_tokens=1024,     # Maximum length of generated text
            ),
        )
    ]
    
    print("✓ Model configuration created")
    
    # Create configuration builder
    config_builder = DataDesignerConfigBuilder(model_configs=model_configs)
    
    print("✓ Config builder initialized\n")
    
    return config_builder, MODEL_ALIAS


# ============================================================
# STEP 3: Define DIMO Data Categories
# ============================================================

def define_dimo_categories():
    """
    Define all DIMO vehicle telemetry categories and subcategories
    
    This creates the structure for what data we want to generate.
    Each category represents a logical grouping of vehicle signals.
    """
    print("="*60)
    print("STEP 3: Defining DIMO Data Categories")
    print("="*60)
    
    # Main categories from DIMO's telemetry schema
    categories = {
        "vehicle_info_status": [
            "odometer_reading",
            "ignition_status", 
            "vehicle_speed",
            "powertrain_type",
            "remaining_range"
        ],
        "location_data": [
            "latitude_longitude",
            "altitude",
            "approximate_location",
            "location_privacy_zones"
        ],
        "battery_charging": [
            "state_of_charge",
            "charging_status",
            "charge_limit",
            "battery_power",
            "charging_current_voltage",
            "gross_battery_capacity",
            "remaining_energy",
            "low_voltage_battery"
        ],
        "engine_metrics": [
            "engine_rpm",
            "throttle_position",
            "engine_air_intake",
            "oil_level",
            "coolant_temperature"
        ],
        "fuel_system": [
            "fuel_type",
            "fuel_percentage",
            "fuel_level_liters"
        ],
        "tire_pressure": [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel"
        ],
        "doors_windows": [
            "front_driver_door",
            "front_passenger_door",
            "rear_driver_door",
            "rear_passenger_door",
            "front_driver_window",
            "front_passenger_window",
            "rear_driver_window",
            "rear_passenger_window"
        ],
        "diagnostics": [
            "diagnostic_trouble_codes",
            "engine_runtime",
            "intake_temperature",
            "engine_load",
            "barometric_pressure"
        ],
        "environmental": [
            "exterior_air_temperature"
        ],
        "device_connectivity": [
            "wifi_status",
            "ssid",
            "gps_satellites",
            "gps_precision"
        ]
    }
    
    # Print summary
    print(f"Total Categories: {len(categories)}")
    for category, subcategories in categories.items():
        print(f"  - {category}: {len(subcategories)} subcategories")
    
    print("✓ Categories defined\n")
    
    return categories


# ============================================================
# STEP 4: Create Sampler Columns
# ============================================================

def create_category_subcategory_columns(config_builder, categories, model_alias):
    """
    Create sampler columns for category and subcategory
    
    NeMo Data Designer uses "samplers" to generate different types of data.
    For DIMO data, we need:
    - A category sampler (picks which main category)
    - A subcategory sampler (picks which subcategory within that category)
    """
    print("="*60)
    print("STEP 4: Creating Sampler Columns")
    print("="*60)
    
    # Get list of all categories
    category_list = list(categories.keys())
    
    # Step 4a: Create CATEGORY column sampler
    print("\n4a. Setting up Category Sampler...")
    config_builder.add_column(
        SamplerColumnConfig(
            name="category",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(
                values=category_list
            )
        )
    )
    print(f"   ✓ Added 'category' column with {len(category_list)} categories")
    
    # Step 4b: Create SUBCATEGORY column sampler
    print("\n4b. Setting up Subcategory Sampler...")
    
    # Build subcategory mapping: each category maps to its subcategories
    subcategory_mapping = {}
    for category, subcategories in categories.items():
        subcategory_mapping[category] = subcategories
    
    config_builder.add_column(
        SamplerColumnConfig(
            name="subcategory",
            sampler_type=SamplerType.SUBCATEGORY,
            params=SubcategorySamplerParams( 
                category="category",
                values=subcategory_mapping 
            )
        )
    )
    
    total_subcategories = sum(len(subs) for subs in categories.values())
    print(f"   ✓ Added 'subcategory' column with {total_subcategories} total subcategories")
    print("   ✓ Subcategories are linked to their parent categories\n")
    
    return config_builder


# ============================================================
# STEP 5: Add Data Value Generation Column
# ============================================================

def add_telemetry_value_column(config_builder, model_alias):
    print("="*60)
    print("STEP 5: Adding Telemetry Value Generation Column")
    print("="*60)
    
    # Use simplified API instead of LLMTextColumnConfig
    config_builder.add_column(
        name="telemetry_value",
        column_type="llm-text",  # String type instead of LLMTextColumnConfig object
        model_alias=model_alias,
        prompt="Generate a realistic telemetry value for {{category}} - {{subcategory}}. Return only the numeric value with units."
    )
    
    print("✓ Added 'telemetry_value' column")
    return config_builder

# ============================================================
# STEP 6: Generate the Dataset
# ============================================================

def generate_dimo_dataset(client, config_builder, num_samples=10):
    print("="*60)
    print("STEP 6: Generating Dataset")
    print("="*60)
    print(f"Generating {num_samples} samples...")
    print("This may take a few moments...\n")
    
    try:
        print("Calling client.preview()...")
        preview_result = client.preview(
            config_builder,
            num_records=num_samples
        )
        
        print(f"✓ Preview completed")
        print(f"Result type: {type(preview_result)}")
        
        # Check different ways to access the data
        dataset = None
        
        if preview_result is None:
            print("❌ preview_result is None!")
            return None
            
        # Try different attribute names
        if hasattr(preview_result, 'dataset'):
            dataset = preview_result.dataset
            print(f"✓ Found .dataset attribute")
        elif hasattr(preview_result, 'data'):
            dataset = preview_result.data
            print(f"✓ Found .data attribute")
        elif hasattr(preview_result, '__iter__'):
            dataset = preview_result
            print(f"✓ Using preview_result directly (iterable)")
        else:
            print(f"⚠ Available attributes: {[attr for attr in dir(preview_result) if not attr.startswith('_')]}")
            
        if dataset is not None:
            print(f"✓ Dataset type: {type(dataset)}")
            print(f"✓ Successfully generated {len(dataset)} samples!")
            print("\nFirst sample:")
            print(dataset.head(1))
        else:
            print("❌ Could not extract dataset from preview_result")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None
    

# ============================================================
# STEP 7: Save and Display Results
# ============================================================

def save_and_display_results(dataset, output_file="dimo_telemetry_data.json"):
    """
    Save the generated dataset and display a preview
    """
    print("\n" + "="*60)
    print("STEP 7: Saving Results")
    print("="*60)
    
    # Check if dataset is None or empty
    if dataset is None:
        print("❌ No dataset to save - dataset is None")
        return
    
    if len(dataset) == 0:
        print("❌ Dataset is empty - no records generated")
        return
    
    # Convert dataset to JSON-serializable format
    data_list = []
    for idx, row in dataset.iterrows():
        record = {
            "sample_id": idx,
            "category": row["category"],
            "subcategory": row["subcategory"]
        }
        
        # Only add telemetry_value if it exists
        if "telemetry_value" in dataset.columns:
            record["telemetry_value"] = row["telemetry_value"]
        
        data_list.append(record)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=2)
    
    print(f"✓ Dataset saved to: {output_file}")
    
    # Display preview
    print(f"\n{'='*60}")
    print("Preview of Generated Data (first 5 samples):")
    print('='*60)
    
    for i, sample in enumerate(data_list[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Category: {sample['category']}")
        print(f"  Subcategory: {sample['subcategory']}")
        if "telemetry_value" in sample:
            print(f"  Value: {sample['telemetry_value']}")
    
    # Statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print('='*60)
    
    category_counts = {}
    for sample in data_list:
        cat = sample['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"Total samples: {len(data_list)}")
    print(f"Unique categories: {len(category_counts)}")
    print("\nSamples per category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} samples")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function - orchestrates all steps
    """
    print("\n" + "="*60)
    print("DIMO VEHICLE TELEMETRY DATA GENERATOR")
    print("Using NVIDIA NeMo Data Designer")
    print("="*60 + "\n")
    
    try:
        # STEP 1: Initialize client
        client = initialize_nemo_client()
        
        # STEP 2: Configure model
        config_builder, model_alias = setup_model_configuration()
        
        # STEP 3: Define categories
        categories = define_dimo_categories()
        
        # STEP 4: Create category/subcategory columns
        config_builder = create_category_subcategory_columns(
            config_builder, categories, model_alias
        )
        
        # STEP 5: Add value generation column
        config_builder = add_telemetry_value_column(config_builder, model_alias)
        
        # STEP 6: Generate dataset
        dataset = generate_dimo_dataset(client, config_builder, num_samples=10)
        
        # STEP 7: Save and display
        save_and_display_results(dataset)
        
        print("\n" + "="*60)
        print("✓ GENERATION COMPLETE!")
        print("="*60)
        print("\nYour DIMO telemetry dataset is ready!")
        print("Check 'dimo_telemetry_data.json' for the full dataset.\n")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure NeMo Microservice is running at http://localhost:8080")
        print("2. Check that nemo_microservices package is installed")
        print("3. Verify your model configuration is correct")
        print("4. Review error message above for specific issues\n")
        raise


if __name__ == "__main__":
    main()