# Azure Resource Setup & GitHub Secrets Guide

Follow these steps to set up the necessary Azure resources and configure your GitHub repository for CI/CD.

## 1. Create Azure Resources

### Azure Container Registry (ACR)
1.  Go to the [Azure Portal](https://portal.azure.com/).
2.  Search for "Container registries" and create a new one.
3.  **Registry Name**: Choose a unique name (e.g., `uniragregistry`).
4.  **SKU**: Basic is sufficient for most cases.
5.  Once created, go to **Access keys** and enable **Admin user**. Note down the `Login server`, `Username`, and `password`.

### Azure App Service (Web App for Containers)
1.  Search for "App Services" and create a new Web App.
2.  **Publish**: Docker Container.
3.  **Operating System**: Linux.
4.  **App Service Plan**: Choose a plan with sufficient resources (Premium V2 or V3 recommended for RAG if not using a cloud model).
5.  **Docker Tab**:
    *   **Options**: Single Container.
    *   **Image Source**: Azure Container Registry.
    *   **Registry**: Select your ACR created above.
    *   **Image**: `uni-rag`.
    *   **Tag**: `latest`.

---

## 2. Configure Environment Variables in Azure

Go to your App Service -> **Configuration** -> **Application settings** and add:
- `OLLAMA_BASE_URL`: `http://localhost:11434` (Note: Use `localhost` since Ollama runs inside the container. **Avoid** `host.docker.internal` in Azure).
- `LLM_MODEL`: e.g., `deepseek-v3.1:671b-cloud`.
- `DB_PATH`: `vector_db`.
- `WEBSITES_PORT`: `8000` (CRITICAL: Tells Azure your app is on port 8000).
- `WEBSITES_CONTAINER_START_TIME_LIMIT`: `600` (Recommended: Prevents timeout while loading models).
- `PORT`: `8000`.

---

## 3. Set Up GitHub Secrets

In your GitHub repository, go to **Settings** -> **Secrets and variables** -> **Actions** and add the following repository secrets:

| Secret Name | Value |
| :--- | :--- |
| `REGISTRY_LOGIN_SERVER` | Your ACR Login server (e.g., `uniragregistry.azurecr.io`) |
| `REGISTRY_USERNAME` | Your ACR Admin username |
| `REGISTRY_PASSWORD` | Your ACR Admin password |
| `AZURE_CREDENTIALS` | JSON output of the Azure Service Principal (see below) |

### Creating `AZURE_CREDENTIALS`
Run this command in your local terminal (with Azure CLI installed) or Azure Cloud Shell:

```bash
az ad sp create-for-rbac --name "myAppActionSp" --role contributor --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP_NAME> --sdk-auth
```
*Replace `<SUBSCRIPTION_ID>` and `<RESOURCE_GROUP_NAME>` with your actual values.*

Copy the entire JSON output and paste it into the `AZURE_CREDENTIALS` secret.

---

## 4. Trigger the Pipeline
Push your changes to the `main` branch, and the GitHub Action will automatically build and deploy your app!
