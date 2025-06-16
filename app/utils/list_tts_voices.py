from google.cloud import texttospeech

def get_available_voices(language_code="es-ES"):
    """Lists available TTS voices for a given language code."""
    client = texttospeech.TextToSpeechClient()
    request = texttospeech.ListVoicesRequest()
    response = client.list_voices(request)
    voices = []
    for voice in response.voices:
        if language_code in voice.language_codes:
            voices.append(voice)
    return voices

# Example usage:
spanish_voices = get_available_voices()
if spanish_voices:
    print("Available Spanish voices:")
    for voice in spanish_voices:
        print(f"  - Name: {voice.name}, Languages: {voice.language_codes}")
else:
    print("No Spanish voices found.")
