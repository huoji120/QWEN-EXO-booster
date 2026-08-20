from qwen_exo_booster.document_categories import DocumentCategoryStore
from qwen_exo_booster.knowledge import KnowledgeRepository


def test_category_store_bootstraps_source_families_and_document_assignments(tmp_path):
    repository = KnowledgeRepository(tmp_path / "knowledge")
    document = repository.upsert(
        "fable.md",
        "---\nsource_kind: boeing_fable5_agent_trajectory\n---\n# Fable\n",
    )
    store = DocumentCategoryStore(tmp_path / "state" / "document-categories.sqlite3")

    store.sync_documents("knowledge", (document,))

    categories = {category["category_id"]: category for category in store.categories()}
    assert categories["agent-trajectories"]["document_count"] == 0
    assert (
        categories["boeing_fable5_agent_trajectory"]["parent_id"]
        == "agent-trajectories"
    )
    assert categories["boeing_fable5_agent_trajectory"]["document_count"] == 1


def test_category_store_allows_user_categories_and_stable_renames(tmp_path):
    store = DocumentCategoryStore(tmp_path / "categories.sqlite3")

    created = store.create("fastapi-routing", "FastAPI 路由", "references")
    updated = store.update(
        "fastapi-routing", title="FastAPI 路由与隐式方法", parent_id="references"
    )

    assert created.category_id == "fastapi-routing"
    assert updated.public_dict() == {
        "category_id": "fastapi-routing",
        "title": "FastAPI 路由与隐式方法",
        "parent_id": "references",
        "origin": "user",
        "document_count": 0,
    }
