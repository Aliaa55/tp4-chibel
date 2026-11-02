package ma.emsi.chibel.tp4chibel;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * Test 3 : routage. Un LM choisit la source (le ContentRetriever) à utiliser pour le RAG.
 */
public class TestRoutage {

    public static void main(String[] args) {
        String llmKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Phase 1 du RAG : ingestion des 2 documents dans 2 EmbeddingStores différents
        EmbeddingStore<TextSegment> embeddingStoreRag = creerEmbeddingStore("rag.pdf", embeddingModel);
        EmbeddingStore<TextSegment> embeddingStoreCouscous = creerEmbeddingStore("couscous.pdf", embeddingModel);

        // Phase 2 du RAG : un ContentRetriever pour chaque document
        ContentRetriever contentRetrieverRag = creerContentRetriever(embeddingStoreRag, embeddingModel);
        ContentRetriever contentRetrieverCouscous = creerContentRetriever(embeddingStoreCouscous, embeddingModel);

        // Description du contenu des sources, en langage naturel (sera lue par le LM)
        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(contentRetrieverRag,
                "Support de cours sur l'IA : le fine-tuning et le RAG (Retrieval Augmented Generation) des modèles de langage");
        descriptions.put(contentRetrieverCouscous,
                "Article sur le couscous : histoire, préparation et variantes de ce plat de la cuisine du Maghreb");

        // Le QueryRouter utilise le LM pour choisir le ou les ContentRetrievers à utiliser
        QueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        conversationAvec(assistant);
    }

    // Phase 1 du RAG pour un document : chargement, découpage en segments,
    // calcul des embeddings et enregistrement dans un magasin d'embeddings en mémoire
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(String nomFichier, EmbeddingModel embeddingModel) {
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument(nomFichier, documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        return embeddingStore;
    }

    // ContentRetriever qui retrouve les 2 segments les plus pertinents, avec un score d'au moins 0,5
    private static ContentRetriever creerContentRetriever(EmbeddingStore<TextSegment> embeddingStore,
                                                          EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    // Conversation avec l'assistant ; taper "fin" pour arrêter
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
    }
}
