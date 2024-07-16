import os
from twilio.rest import Client

account_sid = os.environ['ACc2f36d613d1443ccb90ecc1ea0edd88e']
auth_token = os.environ['727b9345fd84ad1b4ab42840a718f5d6']
client = Client(account_sid, auth_token)

token = client.tokens.create()
print(token.ice_servers)