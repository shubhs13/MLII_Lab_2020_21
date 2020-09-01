# Usage
## Endpoints of the project
All the api are served via a cloud instance and can be found at [API](http://35.232.215.158/api/endpoints).
1. All /DRW endpoints expects a base64 encoded string as input in form of application/json with bs64 as the key of the encoded string.
2. Other endpoints expects a image file in with form-data with file as the key of the payload.
> All the endpoints can entertain upto 1MB of payload beyond that is considered abuse and rejected.
