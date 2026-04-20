# OjuSmart AI Engine - Reference API

## URLs

| Usage      | URL                                          |
|------------|----------------------------------------------|
| Production | https://hordricjr-ojusmartai.hf.space        |
| Local      | http://127.0.0.1:8000                        |
| Swagger UI | https://hordricjr-ojusmartai.hf.space/docs   |

---

## Configuration Spring Boot (application.properties)

ojusmart.ai.base-url=https://hordricjr-ojusmartai.hf.space
ojusmart.ai.connect-timeout=5000
ojusmart.ai.read-timeout=60000

---

## ENDPOINT 1 - Analyse de signature

Methode : POST
URL     : /ai/v1/signatures/analyze
Type    : multipart/form-data

Parametre :
  file  (MultipartFile, obligatoire) - image de signature PNG/JPG/BMP/WEBP

Reponse 200 OK :
{
  "embedding": [0.1234, -0.5678, 0.9012, ...],
  "dimension": 2048
}

Champs reponse :
  embedding  List<Float>  Vecteur de 2048 features (ResNet50 timm)
  dimension  int          Toujours 2048

Codes erreur :
  400  image illisible ou format non supporte
  500  erreur interne

---

## ENDPOINT 2 - Detection d'emotion

Methode : POST
URL     : /ai/v1/emotions/detect
Type    : multipart/form-data

Parametre :
  file  (MultipartFile, obligatoire) - photo contenant au moins un visage

Reponse 200 OK :
{
  "emotion": "happy",
  "confidence": 0.9823
}

Champs reponse :
  emotion     String  libelle de l'emotion detectee
  confidence  float   score entre 0.0 et 1.0

Valeurs possibles pour emotion :
  happy    -> Heureux
  sad      -> Triste
  angry    -> En colere
  fear     -> Peur
  surprise -> Surpris
  disgust  -> Degout
  neutral  -> Neutre

Codes erreur :
  400  aucun visage detecte dans l'image
  400  image illisible ou format non supporte
  500  erreur interne

---

## ENDPOINT 3 - Description d'environnement

Methode : POST
URL     : /ai/v1/environment/describe
Type    : multipart/form-data

Parametre :
  file  (MultipartFile, obligatoire) - photo de l'environnement a decrire

Reponse 200 OK :
{
  "description": "a wooden desk with a computer on top"
}

Champs reponse :
  description  String  phrase en anglais generee par BLIP (5 a 30 tokens)

Codes erreur :
  400  image illisible ou format non supporte
  500  erreur interne

---

## ENDPOINT 4 - Health Check

Methode : GET
URL     : /health

Reponse 200 OK :
{
  "status": "healthy",
  "service": "OjuSmart Stateless AI Engine",
  "version": "1.0.0",
  "device": "cpu"
}

Champs reponse :
  status   String  toujours "healthy" si le service repond
  service  String  nom du service
  version  String  version deployee
  device   String  "cpu", "cuda" ou "mps"

---

## Exemples curl

# Health check
curl https://hordricjr-ojusmartai.hf.space/health

# Analyse signature
curl -X POST https://hordricjr-ojusmartai.hf.space/ai/v1/signatures/analyze -F "file=@signature.png"

# Detection emotion
curl -X POST https://hordricjr-ojusmartai.hf.space/ai/v1/emotions/detect -F "file=@photo.jpg"

# Description environnement
curl -X POST https://hordricjr-ojusmartai.hf.space/ai/v1/environment/describe -F "file=@room.jpg"

---

## DTOs Spring Boot

public record EmotionResponse(String emotion, float confidence) {}
public record EmbeddingResponse(List<Float> embedding, int dimension) {}
public record DescriptionResponse(String description) {}

---

## Bean RestTemplate Spring Boot

@Configuration
public class AiEngineConfig {
    @Value("${ojusmart.ai.base-url}")
    private String baseUrl;

    @Bean
    public RestTemplate aiRestTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5_000);
        factory.setReadTimeout(60_000);
        return new RestTemplate(factory);
    }
}

---

## Client Spring Boot complet

@Service
@RequiredArgsConstructor
public class AiEngineClient {

    private final RestTemplate aiRestTemplate;

    @Value("${ojusmart.ai.base-url}")
    private String baseUrl;

    public EmotionResponse detectEmotion(MultipartFile image) throws IOException {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() { return image.getOriginalFilename(); }
        });
        return aiRestTemplate.postForObject(
            baseUrl + "/ai/v1/emotions/detect",
            new HttpEntity<>(body, multipartHeaders()),
            EmotionResponse.class
        );
    }

    public EmbeddingResponse analyzeSignature(MultipartFile image) throws IOException {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() { return image.getOriginalFilename(); }
        });
        return aiRestTemplate.postForObject(
            baseUrl + "/ai/v1/signatures/analyze",
            new HttpEntity<>(body, multipartHeaders()),
            EmbeddingResponse.class
        );
    }

    public DescriptionResponse describeEnvironment(MultipartFile image) throws IOException {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", new ByteArrayResource(image.getBytes()) {
            @Override public String getFilename() { return image.getOriginalFilename(); }
        });
        return aiRestTemplate.postForObject(
            baseUrl + "/ai/v1/environment/describe",
            new HttpEntity<>(body, multipartHeaders()),
            DescriptionResponse.class
        );
    }

    private HttpHeaders multipartHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return headers;
    }
}

---

## Regles importantes

1. Tous les endpoints acceptent uniquement multipart/form-data (jamais JSON en body)
2. Le champ multipart doit s'appeler exactement : file
3. La description BLIP est toujours en anglais
4. /ai/v1/emotions/detect retourne HTTP 400 si aucun visage n'est detecte
5. Timeout recommande : connect=5s, read=60s (inference PyTorch CPU = 10-30s)
6. Verifier /health avant chaque appel si haute disponibilite requise