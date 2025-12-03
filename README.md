---

# **Project Lucian – Smart Traffic & Pavement Intelligence Backend**

### *Azure Digital Twins • Azure Functions • Azure Maps • Machine Learning*

---

## 📌 **Overview**

Project Lucian is a cloud-based smart-infrastructure backend that models real-time road traffic, predicts congestion trends, and estimates pavement stress/condition using ML.

Using Azure Digital Twins + Azure Functions + Azure Maps, the system provides:

* **Real-time traffic ingestion** (every 5 minutes)
* **Live Digital Twin graph of road segments**
* **Congestion forecasting** using a trained ML model
* **Pavement stress forecasting** using a second ML model
* **Historical export** for analytics
* **APIs** for dashboards, demos, and external tools

This backend is designed for transportation agencies (FDOT-like) who need real-time situational awareness + predictive insights to optimize road maintenance, traffic monitoring, and incident response.

---

## 🏗️ **Architecture Summary**

```
Azure Maps Traffic Flow API
        ↓ (Timer Trigger - 5 min)
Azure Function → traffic/collect
        ↓
Blob Storage (raw JSON snapshots)
        ↓
Blob Trigger → traffic/process
        ↓
Azure Digital Twins (RoadSegment;2)
        ↓
Timer Trigger → traffic/forecast/local-ml
ML Model (Random Forest)
        ↓
SegmentForecast;1 twins
        ↓
Timer Trigger → pavement/aggregate
        ↓
PavementSegment;1 twins
        ↓
Timer Trigger → pavement/forecast/local-ml
PavementForecast;1 twins
```

✔ **Fully automated with no manual refresh needed**
✔ **All models stored in Blob Storage**
✔ **Digital Twins Explorer shows real-time + forecasted graph**

---

## 📦 **Digital Twin Models Implemented**

### **1. RoadSegment (dtmi:fgcu:traffic:RoadSegment;2)**

Real-time traffic updated every 5 minutes.

Properties include:

* latitude, longitude
* segmentId
* roadName
* currentSpeed, freeFlowSpeed
* jamFactor
* delayRatio
* GeoJSON LineString
* lastUpdatedUtc

---

### **2. SegmentForecast (dtmi:fgcu:traffic:SegmentForecast;1)**

ML model predicts future congestion (delayRatio_future).

Properties:

* segmentId
* predictedDelayRatio
* predictedJamFactor (placeholder)
* predictedStressIndex (unused)
* generatedAtUtc
* horizonMinutes (5)
* modelVersion

---

### **3. PavementSegment (dtmi:fgcu:traffic:PavementSegment;1)**

Aggregated historical stress metrics from all collected blobs.

Includes:

* avgJamFactor
* avgDelayRatio
* peakHourJamFactor
* pavementStressIndex
* lastAggregatedUtc

---

### **4. PavementForecast (dtmi:fgcu:traffic:PavementForecast;1)**

ML model predicts near-term pavement conditions.

Predicts:

* predictedPavementStressIndex
* predictedConditionScore (0–100)
* generatedAtUtc
* horizonMinutes
* modelVersion

---

## ⚙️ **Azure Functions Implemented**

### ✅ **1. Traffic Collector (Timer – every 5 minutes)**

**Function:** `collect_fort_myers`

* Samples 100+ points around FGCU
* Snaps to nearest road
* Calls Azure Maps Traffic APIs
* Writes raw JSON snapshots into Blob Storage

---

### ✅ **2. Traffic Processor (Blob Trigger)**

**Function:** `process_fort_myers_blob`

* Reads each traffic snapshot blob
* Normalizes and writes `RoadSegment;2` Digital Twins

---

### ✅ **3. Historical CSV Export (HTTP)**

**Endpoint:** `/api/traffic/history/export?blobs=50`

* Collapses last N snapshots into a CSV
* Used for AutoML training
* Download-ready in browser

---

### ✅ **4. Traffic Forecast (Timer + HTTP)**

**Timer:** runs every **5 minutes**
**Manual Trigger:** `POST /api/traffic/forecast/local-ml`

* Loads ML model from Blob
* Reads all RoadSegment twins
* Predicts future delay ratio
* Upserts `SegmentForecast;1` twins

---

### ✅ **5. Pavement Aggregation (Timer + HTTP)**

**Timer:** every **5 minutes**
**Manual:** GET `/api/pavement/aggregate?blobs=50`

* Reads latest traffic data
* Aggregates jam factor + delay ratio over time
* Upserts `PavementSegment;1` twins

---

### ✅ **6. Pavement Forecast (Timer + HTTP)**

**Timer:** every **5 minutes**
**Manual:** `POST /api/pavement/forecast/local-ml`

* Loads pavement ML model
* Reads latest RoadSegments
* Predicts pavement stress & condition
* Upserts `PavementForecast;1` twins

---

## 🤖 **Machine Learning Models**

Two ML models are trained locally & stored in Azure Blob Storage:

### **Traffic Model:**

* Predicts next 5-minute delay ratio
* RandomForestRegressor via RandomizedSearchCV
* Loaded with `get_ml_model()`

### **Pavement Model:**

* Predicts pavement stress index
* Converts to condition score (0–100)
* Loaded with `get_pavement_model()`

### **Model Storage (App Settings):**

| Setting                    | Purpose                           |
| -------------------------- | --------------------------------- |
| `TRAFFIC_MODEL_CONTAINER`  | Blob container name               |
| `TRAFFIC_MODEL_BLOB`       | `traffic_model.joblib`            |
| `PAVEMENT_MODEL_CONTAINER` | Blob container for pavement model |
| `PAVEMENT_MODEL_BLOB`      | `pavement_model.joblib`           |

Both models are cached in-memory for efficiency.

---

## 🔑 **Environment Variables Required**

### **For Maps ingestion**

```
AZURE_MAPS_SUBSCRIPTION_KEY=
```

### **For Storage**

```
AzureWebJobsStorage=
TRAFFIC_CONTAINER=traffic-flow
```

### **For Digital Twins**

```
ADT_SERVICE_URL=
```

### **For ML Models**

```
TRAFFIC_MODEL_CONTAINER=models
TRAFFIC_MODEL_BLOB=traffic_model.joblib
PAVEMENT_MODEL_CONTAINER=ml-models
PAVEMENT_MODEL_BLOB=pavement_model.joblib
```

---

## 🔥 **API Endpoints (Complete)**

### **Traffic APIs**
needs filled

---

### **Pavement APIs**
needs filled


---

## 🖥️ **Digital Twins Explorer Queries**

### **All Road Segments**

```sql
SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:RoadSegment;2')
```

### **All Congestion Forecasts**

```sql
SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:SegmentForecast;1')
```

### **All Pavement Segments**

```sql
SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:PavementSegment;1')
```

### **All Pavement Forecasts**

```sql
SELECT * FROM digitaltwins t WHERE IS_OF_MODEL(t, 'dtmi:fgcu:traffic:PavementForecast;1')
```

---

## 📁 **Project Structure**

```
lucian-backend/
│
├── azure-functions/
│   └── __init__.py   (all triggers & APIs)
│
├── models/
│   ├── traffic_model.joblib
│   └── pavement_model.joblib
│
├── scripts/
│   ├── prepare_traffic_training.py
│   ├── train_local_automl_model.py
│   ├── prepare_pavement_training.py
│   └── train_pavement_model.py
│
└── README.md
```

---

## 🚀 **Status**

| Component               | Status          |
| ----------------------- | --------------- |
| Azure Maps ingestion    | ✅ Fully working |
| Blob history storage    | ✅ Working       |
| RoadSegment twins       | ✔️ Live updated |
| Traffic forecasting ML  | ✔️ Working      |
| Pavement aggregation    | ✔️ Working      |
| Pavement forecasting ML | ✔️ Working      |
| APIs                    | ✔️ Ready        |

---

## ✔️ **Next Steps**

* Ensure API's are accesible for frontend team
* Add relationship modeling
* Include cost analysis section
