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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.Collections;
import java.util.List;
import java.util.Scanner;

/**
 * Test 4 : pas de RAG. Un QueryRouter personnalisé laisse le LM décider s'il faut utiliser le RAG ou pas.
 */
public class Test3 {

    public static void main(String[] args) {
        String llmKey = System.getenv("GEMINI_KEY");
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Phase 1 du RAG : enregistrement des embeddings du support de cours sur le RAG
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);
        List<TextSegment> segments = DocumentSplitters.recursive(300, 30).split(document);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        // Phase 2 du RAG
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // QueryRouter qui demande au LM si la requête porte sur l'IA ; si ce n'est pas le cas, pas de RAG
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                PromptTemplate template = PromptTemplate.from(
                        "Est-ce que la requête '{{it}}' porte sur l'IA ? "
                        + "Réponds seulement par 'oui', 'non' ou 'peut-être'.");
                Prompt prompt = template.apply(query.text());
                String reponse = model.chat(prompt.text());
                if (reponse.toLowerCase().contains("non")) {
                    // Pas de RAG
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPourEviterRag();

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
