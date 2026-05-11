# Health Kahoot – Question Manager

A healthcare-themed **Kahoot-style** quiz platform for managing **True / False** questions.  
Built with **FastAPI + Uvicorn**, managed by **Poetry**, and backed by either **MongoDB** or **Couchbase** (switchable via environment variable).

---

## Project Structure

```
health-kahoot/
├── pyproject.toml          # Poetry project & dependencies
├── .env.example            # Sample environment variables
├── app/
│   ├── main.py             # FastAPI app, routes, Uvicorn entry
│   ├── db.py               # DB connection layer (JDB_Engine switch)
│   ├── crud.py             # Backend CRUD business logic
│   ├── models.py           # Pydantic schemas
│   ├── templates/
│   │   └── index.html      # Jinja2 web UI (SPA-style)
│   └── static/
│       └── style.css       # Stylesheet
```

### File Responsibilities

| File       | Role |
|------------|------|
| `db.py`    | Reads `JDB_Engine` env var, instantiates the correct async driver (Motor for MongoDB, Couchbase SDK), exposes a uniform `DBEngine` interface |
| `crud.py`  | Pure business logic — create, read, list, search, update, delete questions. Calls `db.py`, never touches drivers directly |
| `main.py`  | FastAPI app with REST endpoints + Jinja2-rendered web UI. Manages DB lifecycle via `lifespan` |
| `models.py`| Pydantic models for validation and serialization |

---

## Quick Start

### 1. Install dependencies

```bash
cd health-kahoot
poetry install
```

### 2. Configure the database

```bash
cp .env.example .env
# Edit .env — set JDB_Engine to "mongodb" or "couchbase"
# Fill in the connection details for your chosen engine
```

### 3. Run the server

```bash
# Option A – Poetry script
poetry run start

# Option B – Direct
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Option C – Python
poetry run python -m app.main
```

Open **http://localhost:8000** in your browser.

---

## Environment Variables

| Variable            | Default                       | Description |
|---------------------|-------------------------------|-------------|
| `JDB_Engine`        | `mongodb`                     | `"mongodb"` or `"couchbase"` |
| `MONGO_URI`         | `mongodb://localhost:27017`   | MongoDB connection string |
| `MONGO_DB`          | `health_kahoot`               | MongoDB database name |
| `COUCHBASE_CONNSTR` | `couchbase://localhost`       | Couchbase connection string |
| `COUCHBASE_USER`    | `Administrator`               | Couchbase username |
| `COUCHBASE_PASS`    | `password`                    | Couchbase password |
| `COUCHBASE_BUCKET`  | `health_kahoot`               | Couchbase bucket name |

---

## REST API

| Method   | Endpoint               | Description |
|----------|------------------------|-------------|
| `GET`    | `/api/questions`       | List & search (query params: `search`, `category`, `difficulty`, `skip`, `limit`) |
| `POST`   | `/api/questions`       | Create a question |
| `GET`    | `/api/questions/{id}`  | Get one question |
| `PUT`    | `/api/questions/{id}`  | Update a question |
| `DELETE` | `/api/questions/{id}`  | Delete a question |

---

## Question Categories

Anatomy · Pharmacology · Pathology · Nursing · Nutrition · Public Health · First Aid · Mental Health · General
