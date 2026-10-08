import json
from audit_semantic_sep import audit
r=audit()
print(json.dumps({'status':r['status'],'failed_checks':r['failed_checks'],
    'checks_completed':len(r['checks']),'pending_count':len(r['pending'])}))
