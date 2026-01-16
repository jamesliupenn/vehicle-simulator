# Vehicle Simulator Using NVIDIA NeMo Microservices

NVIDIA NeMo microservices are an API-first set of tools that consists of a Data Designer, which designs synthetic datasets from scratch or seed using AI models, statistical sampling, and configurable data schemas. In this Vehicle Simulator demo, I'm using NeMo Data Designer alongside data schemas from [DIMO](https://dimo.org) to showcase how AI models can be used to simulate vehicle data for a DIMO developer sandbox.

---

## Prerequisites

- **NVIDIA NGC Account** (free at ngc.nvidia.com)
- **NVIDIA NIM API Key** (free tier at build.nvidia.com)

---

## Setup Steps

### Step 1: Download and Install NGC CLI

Visit **[ngc.nvidia.com](https://ngc.nvidia.com)** and [download the NGC CLI for your operating system](https://ngc.nvidia.com/setup/installers/cli).

---

### Step 2: Generate NGC API Key

1. Generate NGC API Key by clicking on **[Generate API Key](https://ngc.nvidia.com/setup/api-key)**
2. **Copy and save the key** (you won't see it again)

---

### Step 3: Configure NGC CLI

On your terminal, run the configuration command:

```bash
ngc config set
```

**Enter when prompted:**
- API key: `[paste your NGC API key]`
- CLI output format: Your choice, I used `json`
- Org name: `[press Enter]`
- Team name: `[press Enter]`
- Ace name: `[press Enter]`

**Verify:**
```bash
ngc registry image list
```

---

### Step 4: Authenticate Docker with NGC Registry

**Export your NGC API key:**

```bash
export NGC_CLI_API_KEY=<your-ngc-api-key>
```

**Login to the NGC Container Registry:**

```bash
echo $NGC_CLI_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

**Expected output:**
```
Login Succeeded
```

---

### Step 5: Download the NeMo Microservices Container
In your working directory, download a version of `nemo-microservices`:

**Download the quickstart package:**
```bash
ngc registry resource download-version \
  "nvidia/nemo-microservices/nemo-microservices-quickstart:25.12"
```

**Navigate to downloaded directory:**
```bash
cd nemo-microservices-quickstart_v25.12
```

---

### Step 6: Configure Environment Variables

**Get your NIM API key from [build.nvidia.com](https://build.nvidia.com)**

**Set required environment variables:**
```bash
export NEMO_MICROSERVICES_IMAGE_REGISTRY="nvcr.io/nvidia/nemo-microservices"
export NEMO_MICROSERVICES_IMAGE_TAG="25.12"
export NIM_API_KEY="<your-build.nvidia.com-api-key>"
```

**Verify:**
```bash
echo $NEMO_MICROSERVICES_IMAGE_REGISTRY
echo $NEMO_MICROSERVICES_IMAGE_TAG
echo $NIM_API_KEY
```

---

### Step 7: Start NeMo Microservices

**Start the service:**

```bash
docker compose --profile data-designer up --detach --quiet-pull --wait
```

**What this does:**
- `--profile data-designer`: Starts data generation services
- `--detach`: Runs in background
- `--quiet-pull`: Minimal output during image pull
- `--wait`: Waits for services to be healthy

**First run:** Takes about 15 minutes (downloads ~20GB of container images)

**Check container status:**
```bash
docker compose ps
```

All services should show `Up (healthy)` status. I used Docker Desktop for simplicity.

**View logs:**
```bash
docker compose logs -f
```

---

### Step 8: Verify Service Health

**Ping the health endpoint:**

```bash
curl http://localhost:8080/health
```

---

## Data Simulation Steps

This script uses NVIDIA NeMo Data Designer API to create realistic synthetic vehicle telemetry data based on DIMO's data schema. It generates data across multiple categories including battery charging, engine metrics, location data, and more.

---

### Step 0: Prerequisites

- **NeMo Microservice running** at `http://localhost:8080`
  - Follow the [NeMo Microservices Setup Guide](https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/docker-compose.html#ndd-docker-compose)
- **Python 3.8+**
- **nemo_microservices package** installed

---

### Step 1: Install Required Package

```bash
pip install -r requirements.txt
```

### Step 2: Run the Generator

```bash
python dimo_telemetry_generator.py
```

**Output:** `dimo_telemetry_data.json` with synthetic telemetry data

The code initializes a NeMo Client using your credentials and connects to the locally running NeMo Microservice at `localhost:8080`

You can easily set up the LLM model configuration by modifying the following default values:
- **Model**: `nvidia/nemotron-3-nano-30b-a3b`
- **Provider**: `nvidiabuild`
- **Temperature**: `0.25` (more deterministic)
- **Max Tokens**: `1024`

The data definitions can be modified as well depending on your preferences:
- `vehicle_info_status` - Odometer, speed, ignition
- `location_data` - GPS coordinates, altitude
- `battery_charging` - SoC, charging status, capacity
- `engine_metrics` - RPM, throttle, coolant temp
- `fuel_system` - Fuel type, level, percentage
- `tire_pressure` - All four wheels
- `doors_windows` - Status of all doors/windows
- `diagnostics` - DTCs, engine load
- `environmental` - Air temperature
- `device_connectivity` - WiFi, GPS satellites

In the demo, the number of synthetic samples are set to 10 for previews.

### Step 3: Review Results
The output data is saved in `dimo_telemetry_data.json` within the working directory

---

### Generated Data Format

```json
[
  {
    "sample_id": 0,
    "category": "battery_charging",
    "subcategory": "state_of_charge",
    "telemetry_value": "85.3%"
  },
  {
    "sample_id": 1,
    "category": "engine_metrics",
    "subcategory": "engine_rpm",
    "telemetry_value": "2450 RPM"
  }
]
```

---

### Customization

#### Change Number of Samples

Edit line in `main()`:
```python
dataset = generate_dimo_dataset(client, config_builder, num_samples=100)
```

#### Modify Model Parameters

In `setup_model_configuration()`:
```python
inference_parameters=InferenceParameters(
    temperature=0.5,      # Increase for more variety
    top_p=0.9,           # Adjust sampling diversity
    max_tokens=2048,     # Allow longer responses
)
```

#### Add Custom Categories

In `define_dimo_categories()`:
```python
categories = {
    "your_category": [
        "your_subcategory_1",
        "your_subcategory_2"
    ],
    # ... existing categories
}
```

#### Change Output File

In `main()`:
```python
save_and_display_results(dataset, output_file="custom_name.json")
```

---

### Example Output

```
============================================================
DIMO VEHICLE TELEMETRY DATA GENERATOR
Using NVIDIA NeMo Data Designer
============================================================

STEP 1: Initializing NeMo Data Designer Client
✓ Connected to: http://localhost:8080
✓ Client initialized successfully!

STEP 2: Setting up Model Configuration
Model Provider: nvidiabuild
Model ID: nvidia/nemotron-3-nano-30b-a3b
✓ Model configuration created

...

✓ GENERATION COMPLETE!
Your DIMO telemetry dataset is ready!
Check 'dimo_telemetry_data.json' for the full dataset.
```

---

### Dataset Statistics Example

```
Total samples: 100
Unique categories: 10

Samples per category:
  battery_charging: 12 samples
  device_connectivity: 8 samples
  doors_windows: 11 samples
  engine_metrics: 9 samples
  environmental: 7 samples
  fuel_system: 10 samples
  location_data: 13 samples
  tire_pressure: 10 samples
  vehicle_info_status: 12 samples
```

---

### Next Steps

- Modify categories to match your specific use case
- Integrate with DIMO protocol for data validation
- Export to different formats (CSV, Parquet)
- Scale up generation for larger datasets
- Add validation rules for telemetry values
