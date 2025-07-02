from prometheus_client import Summary

metrics_labels = ['app_name', 'method', 'http_status', 'req_url']

app_metrics = {}

app_metrics['request_time'] = Summary(
    'request_processing_seconds', 
    'Time spent processing request', 
    metrics_labels
    )