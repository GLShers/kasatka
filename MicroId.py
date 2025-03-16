import sounddevice as sd

def list_microphones():
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        # Проверяем, поддерживает ли устройство вход (микрофон)
        if device['max_input_channels'] > 0:
            input_devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels']
            })
    
    return input_devices

if __name__ == "__main__":
    mics = list_microphones()
    
    if not mics:
        print("Микрофоны не найдены!")
    else:
        print("Доступные микрофоны:")
        for mic in mics:
            print(f"ID: {mic['id']}, Название: {mic['name']}, Каналы: {mic['channels']}")