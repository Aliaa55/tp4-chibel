package ma.emsi.chibel.tp4chibel;

/**
 * Assistant conversationnel utilisé par les tests sur le RAG.
 * L'implémentation est faite par LangChain4j (AiServices).
 */
public interface Assistant {
    // Prend un message de l'utilisateur et retourne une réponse du LLM.
    String chat(String userMessage);
}
