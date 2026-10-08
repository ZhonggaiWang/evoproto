import json
from audit_prototype_sep import audit
obj=audit()
print(json.dumps({'status':obj['status'],'failed_checks':obj['failed_checks'],
    'pending':obj['pending'],'checked':len(obj['checks'])+sum(len(s['checks']) for s in obj['stages'])}))
