# RP-kullish 
ETL PIPELINES STANDARDIZATION

### Workflows--ML Pipeline

1. Data Ingestion
2. Data Validation
3. Data Transformation-- Feature Engineering,Data Preprocessing
4. Model Trainer
5. Model Evaluation- MLFLOW,Dagshub

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py

## Dynamic Data Ingestion

The data ingestion pipeline now supports dynamic source URLs. Users can easily specify custom URLs without modifying the configuration files.

### Supported URL Types:
- **Direct URLs**: `https://example.com/data.zip`
- **GitHub Raw URLs**: `https://raw.githubusercontent.com/user/repo/main/data.csv`
- **Google Drive URLs**: `https://drive.google.com/file/d/FILE_ID/view`

### Method 1: Environment Variable
Set the `SOURCE_URL` environment variable:
```bash
export SOURCE_URL="https://your-custom-url.com/data.zip"
python main.py
```

### Method 2: Command Line Argument
Use the provided runner script:
```bash
python run_data_ingestion.py --url "https://your-custom-url.com/data.zip"
```

### Method 3: .env File
Create a `.env` file in the project root:
```
SOURCE_URL=https://your-custom-url.com/data.zip
```
Then run:
```bash
python run_data_ingestion.py --env-file .env
```

### Google Drive Support
The system automatically converts Google Drive sharing URLs to direct download URLs:

**Supported Google Drive URL Formats:**
- `https://drive.google.com/file/d/FILE_ID/view`
- `https://drive.google.com/open?id=FILE_ID`
- `https://docs.google.com/uc?export=download&id=FILE_ID`

**Example:**
```bash
export SOURCE_URL="https://drive.google.com/file/d/1ABC123xyz/view"
python main.py
```

### Default Behavior
If no custom URL is provided, the system will use the default URL specified in `config/config.yaml`.

## Dynamic Schema Generation

The system now automatically generates schema based on the actual data structure, eliminating the need for manual schema configuration.

### Features:
- **Automatic Type Detection**: Infers column data types from the data
- **Target Column Suggestion**: Intelligently suggests the target column
- **Multi-format Support**: Works with CSV, JSON, Parquet, Excel files
- **Real-time Updates**: Schema.yaml is automatically updated after data ingestion

### How It Works:
1. Data is downloaded and extracted
2. System analyzes the data structure
3. Column types are automatically inferred
4. Target column is suggested based on common patterns
5. Schema.yaml is updated with the new structure

### Supported Data Types:
- **Numeric**: int64, float64
- **Categorical**: category (auto-detected for low-cardinality columns)
- **Text**: object (for high-cardinality text columns)
- **Boolean**: bool
- **DateTime**: datetime64

### Target Column Detection:
The system looks for common target column names:
- `target`, `label`, `class`, `category`, `quality`, `rating`
- `score`, `outcome`, `result`, `prediction`, `y`, `target_variable`
- Falls back to the last column if no obvious target is found


### urls for testing
1. https://drive.google.com/file/d/1vhihD-30SLa_clbNwmvqsl_boZDfB7q6/view?usp=sharing
2.https://github.com/Rohan-doodleblue/datasets/blob/main/winequality-data.zip
