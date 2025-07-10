from tapo import ApiClient

p100 = PyP100.P100("192.168.0.167", "desmondeds@hotmail.com", "xxxxxx&2014")  # Creates a P100 plug object

client = ApiClient("<tapo-username>", "tapo-password")
device = await client.p110("<device ip address>")

await device.on()