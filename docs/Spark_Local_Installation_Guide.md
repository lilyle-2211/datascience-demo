# Apache Spark Local Installation Guide

## What's New in Spark 4.0

### Major Changes
- **Java 8 Support Dropped:** Minimum Java 11 required
- **Python 3.8+ Required:** Python 3.7 and earlier no longer supported
- **Scala 2.13 Default:** Scala 2.12 still supported but 2.13 is default
- **Performance Improvements:** Better query optimization and execution
- **New SQL Features:** Enhanced ANSI SQL compliance
- **Improved Streaming:** Better fault tolerance and performance

### Breaking Changes
- Some deprecated APIs removed
- Configuration property changes
- Behavior changes in certain SQL functions

## Prerequisites

### Java Installation
Spark 4.0 requires Java 11, 17, or 21 (Java 8 is no longer supported).

#### Check Java Version:
```bash
java -version
javac -version
```

#### Install Java (if needed):

**macOS:**
```bash
# Using Homebrew (Java 17 recommended for Spark 4.0)
brew install openjdk@17

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"

# Alternative: Java 21
brew install openjdk@21
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"
```

**Ubuntu/Linux:**
```bash
sudo apt update
# Install Java 17 (recommended)
sudo apt install openjdk-17-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Alternative: Java 21
sudo apt install openjdk-21-jdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

**Windows:**
- Download from [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://openjdk.org/)
- Set `JAVA_HOME` environment variable

## Method 1: Download and Install Manually

### Step 1: Download Spark
```bash
# Download Spark 4.0 (latest version)
cd ~/Downloads
wget https://downloads.apache.org/spark/spark-4.0.0/spark-4.0.0-bin-hadoop3.tgz

# Or use curl
curl -O https://downloads.apache.org/spark/spark-4.0.0/spark-4.0.0-bin-hadoop3.tgz
```

### Step 2: Extract and Install
```bash
# Extract
tar -xzf spark-4.0.0-bin-hadoop3.tgz

# Move to installation directory
sudo mv spark-4.0.0-bin-hadoop3 /opt/spark

# Create symlink (optional)
sudo ln -s /opt/spark /usr/local/spark
```

### Step 3: Set Environment Variables
Add to `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`:

```bash
# Spark Environment
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python3
```

### Step 4: Apply Changes
```bash
source ~/.zshrc  # or ~/.bashrc
```

### Add to your ~/.zshrc (or ~/.bashrc) for persistent setup:

```bash
# Java and Spark environment for Apache Spark
export JAVA_HOME="/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export SPARK_HOME="/opt/spark"
export PATH="$JAVA_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH"
export PYSPARK_PYTHON=python3
```

After editing, apply changes with:
```bash
source ~/.zshrc
```

Now, Java and Spark will always be available in new terminal sessions.

## Method 2: Using Package Managers

### macOS with Homebrew
```bash
# Install Spark
brew install apache-spark

# Spark will be installed to /opt/homebrew/Cellar/apache-spark/
# Environment variables are set automatically
```

### Ubuntu/Linux with Package Manager
```bash
# Add Spark repository
wget -qO - https://downloads.apache.org/spark/KEYS | sudo apt-key add -
echo "deb https://downloads.apache.org/spark/ /" | sudo tee /etc/apt/sources.list.d/spark.list

sudo apt update
sudo apt install spark
```

## Method 3: Using Python pip (PySpark Only)

### Install PySpark
```bash
# Install PySpark
pip install pyspark

# Or with specific version
pip install pyspark==4.0.0
```

### Verify Installation
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TestApp") \
    .master("local[*]") \
    .getOrCreate()

print(f"Spark version: {spark.version}")
spark.stop()
```

## Verification and Testing

### Test Spark Installation

#### 1. Check Spark Version
```bash
spark-submit --version
```

#### 2. Start Spark Shell
```bash
# Scala Shell
spark-shell

# Python Shell
pyspark

# SQL Shell  
spark-sql
```

#### 3. Test with Sample Data
```bash
# Run example application
spark-submit \
  --class org.apache.spark.examples.SparkPi \
  $SPARK_HOME/examples/jars/spark-examples*.jar \
  10
```

### Test PySpark Script
Create `test_spark.py`:
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Create Spark session
spark = SparkSession.builder \
    .appName("LocalSparkTest") \
    .master("local[2]") \
    .getOrCreate()

# Create sample data
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

# Show DataFrame
print("Sample DataFrame:")
df.show()

# Perform operations
print("Count:", df.count())
print("Schema:")
df.printSchema()

# Stop Spark session
spark.stop()
print("Spark session stopped successfully!")
```

Run the test:
```bash
python test_spark.py
```

## Spark Configuration

### Basic Configuration File
Create `$SPARK_HOME/conf/spark-defaults.conf`:
```
# Basic Spark configuration
spark.master                     local[*]
spark.driver.memory              2g
spark.executor.memory            2g
spark.sql.warehouse.dir          file:///tmp/spark-warehouse
spark.serializer                 org.apache.spark.serializer.KryoSerializer
```

### Environment Configuration
Create `$SPARK_HOME/conf/spark-env.sh`:
```bash
#!/usr/bin/env bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64  # Adjust path for your system
export PYSPARK_PYTHON=python3.8  # Python 3.8+ required for Spark 4.0
export SPARK_LOCAL_DIRS=/tmp/spark-temp

# For macOS with Homebrew Java
# export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
```

Make executable:
```bash
chmod +x $SPARK_HOME/conf/spark-env.sh
```

## Common Issues and Solutions

### Issue 1: Java Not Found
**Error:** `java: command not found`

**Solution:**
```bash
# Check Java installation
which java
echo $JAVA_HOME

# Install Java if missing
# macOS:
brew install openjdk@11

# Ubuntu:
sudo apt install openjdk-11-jdk
```

### Issue 2: Permission Denied
**Error:** Permission issues with Spark directories

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER:$USER /opt/spark
chmod +x $SPARK_HOME/bin/*
```

### Issue 3: Python Path Issues
**Error:** PySpark can't find Python

**Solution:**
```bash
# Set Python path explicitly
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Or use specific Python version
export PYSPARK_PYTHON=/usr/bin/python3.9
```

### Issue 4: Memory Issues
**Error:** `OutOfMemoryError`

**Solution:**
```bash
# Increase driver memory
spark-submit --driver-memory 4g your_script.py

# Or set in configuration
export SPARK_DRIVER_OPTS="-Xmx4g"
```

### Issue 5: Port Already in Use
**Error:** `Address already in use`

**Solution:**
```bash
# Check what's using port 4040
lsof -i :4040

# Use different port
spark-submit --conf spark.ui.port=4041 your_script.py
```

## Spark UI and Monitoring

### Access Spark Web UI
- **Driver UI:** http://localhost:4040 (default)
- **History Server:** http://localhost:18080

### Start History Server
```bash
# Start Spark history server
$SPARK_HOME/sbin/start-history-server.sh

# Stop history server
$SPARK_HOME/sbin/stop-history-server.sh
```

## Development Setup

### IDE Configuration

#### VS Code Setup
1. Install Python extension
2. Install Scala extension (if using Scala)
3. Configure Python interpreter to include PySpark

#### Jupyter Notebook Setup
```bash
# Install Jupyter
pip install jupyter

# Set environment variables for Jupyter
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

# Start PySpark with Jupyter
pyspark
```

### Sample Jupyter Cell
```python
# Initialize Spark in Jupyter
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("JupyterSpark") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Your Spark code here
df = spark.range(1000).toDF("id")
df.show(10)
```

## Performance Tuning for Local Development

### Optimize for Local Machine
```bash
# Set appropriate number of cores
export SPARK_LOCAL_CORES=4

# Adjust memory settings
export SPARK_DRIVER_MEMORY=2g
export SPARK_EXECUTOR_MEMORY=2g

# Enable adaptive query execution
export SPARK_SQL_ADAPTIVE_ENABLED=true
```

### Sample Local Configuration
Create `spark_local.conf`:
```
spark.master                     local[4]
spark.driver.memory              2g
spark.executor.memory            2g
spark.sql.adaptive.enabled       true
spark.sql.adaptive.coalescePartitions.enabled  true
spark.serializer                 org.apache.spark.serializer.KryoSerializer
spark.kryo.unsafe                true
```

Use configuration:
```bash
spark-submit --properties-file spark_local.conf your_script.py
```

## Next Steps

### Learning Resources
- **Official Documentation:** https://spark.apache.org/docs/latest/
- **PySpark Documentation:** https://spark.apache.org/docs/latest/api/python/
- **Spark SQL Guide:** https://spark.apache.org/docs/latest/sql-programming-guide.html

### Sample Projects
1. **Data Processing:** ETL pipelines with CSV/JSON files
2. **Machine Learning:** MLlib for classification/regression
3. **Streaming:** Structured streaming with file sources
4. **SQL Analytics:** Complex aggregations and window functions

### Production Considerations
- **Cluster Deployment:** YARN, Kubernetes, Standalone
- **Resource Management:** Dynamic allocation
- **Monitoring:** Spark History Server, metrics
- **Security:** Authentication and authorization

## Migrating from Spark 3.x to 4.0

### Pre-Migration Checklist
1. **Upgrade Java:** Ensure Java 11+ (Java 8 no longer supported)
2. **Upgrade Python:** Ensure Python 3.8+ (Python 3.7 and earlier deprecated)  
3. **Review Dependencies:** Check if your libraries support Spark 4.0
4. **Test Applications:** Run tests against Spark 4.0 in development

### Common Migration Issues
```bash
# Issue: Java 8 no longer supported
# Solution: Upgrade to Java 17
brew install openjdk@17
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"

# Issue: Python 3.7 deprecated  
# Solution: Upgrade to Python 3.9+
pyenv install 3.11.0
pyenv global 3.11.0

# Issue: Old PySpark version
# Solution: Upgrade PySpark
pip uninstall pyspark
pip install pyspark==4.0.0
```

### Configuration Updates
Some configuration properties may need updates:
```bash
# Old Spark 3.x configurations that may need review
spark.sql.adaptive.enabled=true  # Still valid
spark.sql.adaptive.coalescePartitions.enabled=true  # Still valid

# Check Spark 4.0 migration guide for deprecated configs
# https://spark.apache.org/docs/4.0.0/migration-guide.html
```

### Compatibility Testing
```python
# Test script to verify Spark 4.0 compatibility
from pyspark.sql import SparkSession
import sys

print(f"Python version: {sys.version}")

spark = SparkSession.builder \
    .appName("Spark4CompatibilityTest") \
    .master("local[2]") \
    .getOrCreate()

print(f"Spark version: {spark.version}")
print(f"Java version: {spark.sparkContext._jvm.System.getProperty('java.version')}")

# Test basic operations
df = spark.range(100).toDF("id")
result = df.filter(df.id > 50).count()
print(f"Test query result: {result}")

spark.stop()
print("✅ Spark 4.0 compatibility test passed!")
```

## Summary

Local Spark installation options:
1. **Manual Installation:** Full control, all components
2. **Package Managers:** Easy installation, automatic updates  
3. **PySpark pip:** Python-only, simplest for data science

Choose based on your needs:
- **Full Spark ecosystem:** Manual installation
- **Python development only:** PySpark pip
- **macOS users:** Homebrew recommended