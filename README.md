# AFRICA-LMM

### Système multimodal open source de compréhension documentaire et de Retrieval-Augmented Generation pour les documents africains

**AFRICA-LMM** est un système d'intelligence artificielle multimodal open source conçu pour comprendre, rechercher et interroger des documents africains.

La plateforme combine traitement documentaire, OCR, vision par ordinateur, embeddings, recherche vectorielle, reranking, inférence LLM, citations et évaluation dans une architecture complète d'ingénierie Machine Learning.

Le projet cible notamment les documents liés à l'agriculture, l'éducation, l'économie, le climat et les rapports publics, avec un support initial du français et de l'anglais.

---

## Pourquoi AFRICA-LMM ?

Une grande partie de l'information africaine est distribuée dans des PDF, des documents scannés, des tableaux et des rapports contenant des images.

Une recherche classique par mots-clés est souvent insuffisante pour exploiter correctement ce type de contenu.

AFRICA-LMM construit une chaîne complète :

```text
PDF / Image / Texte
       │
       ▼
Traitement documentaire
       │
       ├── Extraction PDF
       ├── OCR
       ├── Tableaux
       └── Images
       │
       ▼
Découpage en chunks
       │
       ▼
Embeddings
       │
       ▼
Base vectorielle Qdrant
       │
       ▼
Recherche sémantique
       │
       ▼
Cross-Encoder Reranking
       │
       ▼
Construction du contexte
       │
       ▼
Inférence LLM
       │
       ▼
Réponse + preuves + citations
```

---

## Fonctionnalités principales

* Ingestion multimodale de documents
* Extraction de texte depuis les PDF
* OCR avec Tesseract
* OCR français et anglais
* Extraction de tableaux
* Traitement d'images
* Embeddings sémantiques
* Recherche vectorielle avec Qdrant
* Reranking avec Cross-Encoder
* Retrieval-Augmented Generation (RAG)
* Citations au niveau des sources et des pages
* Gestion des conversations
* Chargement paresseux des modèles
* Fine-tuning avec LoRA
* Pipeline QLoRA
* Framework d'évaluation et de benchmark
* API d'inférence FastAPI
* Intégration PostgreSQL
* Déploiement Docker
* Interface web Next.js
* CI automatisée avec GitHub Actions

---

## Architecture

```text
                         ┌─────────────────────┐
                         │      Interface      │
                         │       Next.js       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │        API          │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
        ┌──────────────────┐                ┌──────────────────┐
        │ Moteur de        │                │ Services de      │
        │ documents        │                │ conversation     │
        └────────┬─────────┘                └──────────────────┘
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
      OCR      Vision    Tableaux
        │        │         │
        └────────┼─────────┘
                 ▼
        ┌──────────────────┐
        │    Chunking      │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │    Embeddings    │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │      Qdrant      │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │    Retrieval     │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │    Reranking     │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Construction     │
        │ du contexte      │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │     LLM / VLM    │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Réponse +        │
        │ sources          │
        └──────────────────┘
```

---

## Structure du dépôt

```text
africa-lmm/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── datasets/
│   ├── experiments/
│   ├── rag/
│   └── training/
│
├── data/
│   ├── datasets/
│   └── test_documents/
│
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── database/
│   ├── evaluation/
│   ├── inference/
│   ├── ocr/
│   ├── rag/
│   ├── training/
│   └── vision/
│
├── tests/
│   ├── benchmarks/
│   ├── integration/
│   └── unit/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Pipeline RAG

Le cœur du système suit le flux suivant :

```text
Question utilisateur
       │
       ▼
Routage de l'intention
       │
       ▼
Embedding de la requête
       │
       ▼
Recherche vectorielle
       │
       ▼
Documents candidats
       │
       ▼
Cross-Encoder Reranking
       │
       ▼
Construction du contexte
       │
       ▼
Stratégie de réponse
       │
       ▼
LLM
       │
       ▼
Réponse fondée sur les preuves
       +
Citations
```

Le système ne repose donc pas uniquement sur la génération du LLM : les preuves récupérées sont conservées tout au long du pipeline et retournées avec la réponse.

---

## Traitement multimodal des documents

AFRICA-LMM peut combiner plusieurs sources d'information provenant d'un même document :

```text
PDF
 │
 ├── Extraction de texte natif
 │
 ├── Détection des pages scannées
 │        │
 │        └── OCR
 │
 ├── Images intégrées
 │
 └── Tableaux
```

Le module OCR utilise actuellement Tesseract avec le français et l'anglais.

Le module de vision prend notamment en charge :

```text
JPEG
PNG
WEBP
TIFF
BMP
```

---

## Modèles

La couche d'inférence repose sur l'écosystème Hugging Face Transformers.

Configuration actuelle de développement :

```text
LLM :
Qwen/Qwen2.5-0.5B-Instruct

Modèle d'embeddings :
Sentence Transformers

Reranker :
BAAI/bge-reranker-v2-m3
```

Les modèles sont chargés de manière paresseuse lorsque cela est possible afin de réduire la consommation mémoire au démarrage et d'éviter les téléchargements inutiles.

---

## Fine-tuning avec LoRA

AFRICA-LMM contient des pipelines dédiés au fine-tuning LoRA et QLoRA.

Une expérimentation LoRA a été exécutée pendant **3 epochs**.

### Résultats mesurés

| Métrique            | Résultat |
| ------------------- | -------: |
| Epochs              |        3 |
| Training loss       |    3.027 |
| Entropy             |    1.988 |
| Mean token accuracy |  59.58 % |

Checkpoint :

```text
data/checkpoints/africa-lmm-lora
```

Le code d'entraînement est séparé du pipeline d'inférence et du système RAG afin de permettre l'évolution indépendante des modèles et de l'infrastructure de serving.

---

## Évaluation du système RAG

Un benchmark reproductible a été exécuté sur **10 éléments d'évaluation**.

### Résultats mesurés

| Métrique              | Résultat |
| --------------------- | -------: |
| Retrieval Recall@1    |     90 % |
| Retrieval Recall@5    |     90 % |
| Reranker Recall@1     |     90 % |
| Answer Accuracy       |     70 % |
| Citation Accuracy     |     90 % |
| Unanswerable Accuracy |      0 % |

Ces valeurs correspondent à des **résultats mesurés pendant le développement** et non à des objectifs théoriques.

Les artefacts du benchmark sont disponibles dans :

```text
tests/benchmarks/
```

MLflow est utilisé pour le suivi des expériences et des évaluations.

### Limitation identifiée

La détection des questions auxquelles la base documentaire ne permet pas de répondre correctement reste un point d'amélioration important.

Le résultat de `Unanswerable Accuracy = 0 %` est donc volontairement conservé dans la documentation afin de rendre visibles les limites actuelles du système.

---

## API

Le backend est développé avec FastAPI.

Principales routes :

```text
GET  /health

POST /query

POST /documents

GET  /documents

GET  /documents/{document_id}/file

POST /conversations

GET  /conversations

GET  /conversations/{conversation_id}

DELETE /conversations/{conversation_id}
```

FastAPI fournit également automatiquement la documentation OpenAPI lorsque le serveur est lancé.

---

## Lancer le projet avec Docker

### Prérequis

* Docker
* Docker Compose
* 8 Go de RAM minimum recommandés pour l'environnement de développement
* Connexion Internet pour le téléchargement initial des modèles

Cloner le dépôt :

```bash
git clone https://github.com/le-roi-y/africa-lmm.git
cd africa-lmm
```

Configurer l'environnement :

```bash
cp .env.example .env
```

Démarrer les services :

```bash
docker compose up -d
```

Vérifier l'API :

```bash
curl http://localhost:8001/health
```

### Services

| Service     | Port |
| ----------- | ---: |
| FastAPI     | 8001 |
| Qdrant HTTP | 6333 |
| Qdrant gRPC | 6334 |
| PostgreSQL  | 5432 |

---

## Interface web

Le projet possède une interface frontend développée avec Next.js.

```bash
cd frontend

npm install

printf 'NEXT_PUBLIC_API_URL=http://127.0.0.1:8001\n' > .env.local

npm run dev
```

Puis ouvrir :

```text
http://localhost:3000
```

L'interface permet notamment :

* d'importer des documents ;
* d'indexer les documents ;
* de poser des questions ;
* d'afficher les réponses générées ;
* d'exploiter les citations et sources du système RAG.

---

## Installation pour le développement

Créer l'environnement Python :

```bash
python3.13 -m venv env
source env/bin/activate
```

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

Vérifier la qualité du code :

```bash
ruff check src tests
black --check src tests
```

Lancer les tests :

```bash
pytest -q
```

Vérifier la compilation :

```bash
python -m compileall -q src
```

---

## CI/CD

Le projet utilise GitHub Actions pour automatiser les contrôles principaux :

```text
Checkout
   │
   ▼
Python 3.13
   │
   ▼
Installation des dépendances
   │
   ├── Compilation Python
   ├── Ruff
   ├── Black
   └── Pytest
```

Workflow :

```text
.github/workflows/ci.yml
```

---

## Stack technique

### Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* PEFT
* LoRA
* QLoRA
* Safetensors

### Document AI

* PyMuPDF
* pypdf
* pdfplumber
* Tesseract OCR
* Pillow

### Retrieval

* Qdrant
* Embeddings denses
* Cross-Encoder
* RAG

### Backend

* FastAPI
* Pydantic
* SQLAlchemy
* PostgreSQL

### MLOps

* MLflow
* GitHub Actions
* Docker
* Docker Compose

### Frontend

* Next.js
* React
* TypeScript

---

## Principes d'ingénierie

AFRICA-LMM suit plusieurs principes :

* Architecture modulaire
* Chargement paresseux des modèles
* Injection de dépendances lorsque pertinente
* Composants testables indépendamment
* Évaluation reproductible
* Configuration séparée du code
* Aucun secret versionné
* Services conteneurisés avec Docker
* Contrôles de qualité automatisés
* Réponses accompagnées de preuves et de citations
* Résultats expérimentaux mesurés plutôt que revendiqués

---

## Feuille de route

### Implémenté

* [x] Ingestion PDF
* [x] OCR
* [x] Traitement d'images
* [x] Extraction de tableaux
* [x] Embeddings sémantiques
* [x] Recherche Qdrant
* [x] Reranking Cross-Encoder
* [x] Pipeline RAG
* [x] Citations
* [x] API FastAPI
* [x] Intégration PostgreSQL
* [x] Déploiement Docker
* [x] Frontend Next.js
* [x] Pipeline LoRA
* [x] Pipeline QLoRA
* [x] Benchmark d'évaluation
* [x] Suivi MLflow
* [x] CI GitHub Actions

### Prochaines étapes

* [ ] Intégration VLM multimodale plus avancée
* [ ] Datasets de langues africaines
* [ ] Amélioration de la détection des questions sans réponse
* [ ] Hybrid Retrieval lexical + dense
* [ ] Observabilité du pipeline RAG
* [ ] Monitoring en production
* [ ] Évaluation à plus grande échelle
* [ ] Model Card
* [ ] Dataset Card
* [ ] Démonstration publique

---

## Statut du projet

AFRICA-LMM est un projet open source actif d'ingénierie Machine Learning.

L'objectif actuel est de construire une chaîne complète et reproductible allant de l'ingestion documentaire jusqu'au retrieval, à la génération, à l'évaluation et au serving.

Le projet documente volontairement ses résultats mesurés ainsi que ses limitations afin de présenter une évaluation technique transparente.

---

## Licence

Ce projet est distribué sous licence MIT.

Voir [LICENSE](LICENSE).
