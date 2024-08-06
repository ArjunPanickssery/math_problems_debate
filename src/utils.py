def config(instance):
    result = {}
    if hasattr(instance, '__class__') and hasattr(instance.__class__, '__name__'):
        result['class'] = instance.__class__.__name__
    for key, value in instance.__dict__.items():
        if hasattr(value, '__dict__'):
            result[key] = config(value)
        elif isinstance(value, list):
            result[key] = [config(item) if hasattr(item, '__dict__') else item for item in value]
        elif isinstance(value, dict):
            result[key] = {k: config(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
        else:
            result[key] = value
    return result