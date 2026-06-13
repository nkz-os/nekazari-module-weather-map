from unittest.mock import AsyncMock, patch

import pytest

from app.sources import upsert_record


@pytest.mark.asyncio
async def test_upsert_record_uses_sdk_orionclient():
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    mock.create_entity = AsyncMock(return_value={"id": "x"})
    with patch("app.sources.OrionClient", return_value=mock):
        await upsert_record("asociacion-allotarra", {"id": "urn:x", "type": "AgriParcelRecord"})
    mock.create_entity.assert_awaited_once()
