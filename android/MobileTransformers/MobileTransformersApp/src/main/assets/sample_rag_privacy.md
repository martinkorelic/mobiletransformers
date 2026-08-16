# Privacy, consent, and what leaves the device

## The default

Nothing leaves the phone. Generation, retrieval, ingestion and training all run locally against files
in the app's own storage. There is no inference server, and no prompt or document is transmitted
anywhere as part of normal use. The only network traffic is downloading a model package.

## Federated learning

Federated learning is the one feature that sends anything out, and it sends the smallest possible
thing: the adapter factors produced by local training, plus aggregate metrics such as the number of
examples seen and the average loss. **Training examples never leave the device.** The documents you
ingested, the prompts you typed and the dataset you trained on all stay local; what travels is a set
of small matrices describing how the model changed.

## Consent is checked before any tensor is read

Federation is disabled by default and must be switched on deliberately by the application that ships
the SDK. Beyond that build-time switch, a round refuses to start unless consent has been granted, the
gateway address uses TLS, and an authentication token is present. Those three checks happen before any
adapter tensor is opened, so a misconfigured round fails without ever having touched the weights.

## Rounds

A round imports the current global adapter, trains locally on this device's own data, and exports the
difference. Round zero imports nothing, because a device has to be able to join a group that has not
published an aggregate yet. The round returns the bytes it would upload rather than uploading them
itself — handing them to a gateway is a separate, deliberate act by the application.

## Aggregation

The server averages the updates it receives across participating devices, weighted by how many
examples each one trained on. No individual device's contribution is stored as such, and a device that
drops out mid-round is simply absent from that average rather than blocking it.
