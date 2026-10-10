import time
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for _ in range(60):
    try:
        d=[d for d in api.dataset_list(mine=True) if d.ref.endswith('medleaf-esa-cache')][0]
        if (d.total_bytes or 0)>0: print('READY', d.total_bytes); break
    except Exception as e: print('err', e)
    time.sleep(20)
