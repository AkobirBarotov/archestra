import { CohereClient } from "cohere-ai";
import { get } from "lodash";
import { config } from "../../../config";
import { logger } from "../../../logging";
import {
  LLMProvider,
  LLMRequestAdapter,
  LLMResponseAdapter,
  LLMStreamAdapter,
  ChunkProcessingResult,
  StreamAccumulatorState,
  UsageView,
  CommonToolCall,
} from "../../../types/llm-provider";
import {
  OpenAiRequest,
  OpenAiResponse,
  OpenAiMessages,
  OpenAiStreamChunk,
  OpenAiHeaders,
} from "../../../types/llm-providers/openai/api";

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function convertMessagesToCohere(messages: OpenAiMessages) {
  // Extract the last user message as the "message" parameter
  const lastMessage = messages[messages.length - 1];
  const message = lastMessage?.content && typeof lastMessage.content === 'string' 
    ? lastMessage.content 
    : " "; // Fallback

  // Convert previous messages to chat_history
  const chatHistory = messages.slice(0, -1).map((msg) => {
    let role = "USER";
    if (msg.role === "assistant") role = "CHATBOT";
    if (msg.role === "system") role = "SYSTEM";
    
    // Simple content extraction (ignoring complex image blocks for now)
    let content = "";
    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      content = msg.content
        .filter((c) => c.type === "text")
        .map((c: any) => c.text)
        .join("\n");
    }

    return {
      role: role as "USER" | "CHATBOT" | "SYSTEM",
      message: content,
    };
  });

  return { message, chatHistory };
}

// =============================================================================
// REQUEST ADAPTER
// =============================================================================

class CohereRequestAdapter implements LLMRequestAdapter<OpenAiRequest, OpenAiMessages> {
  private request: OpenAiRequest;

  constructor(request: OpenAiRequest) {
    this.request = request;
  }

  getMessages(): OpenAiMessages {
    return this.request.messages;
  }

  toProviderRequest(): OpenAiRequest {
    // Cohere adapter uses the messages internally in execute/executeStream
    // We return the request as-is for the interface
    return this.request;
  }
}

// =============================================================================
// RESPONSE ADAPTER
// =============================================================================

class CohereResponseAdapter implements LLMResponseAdapter<OpenAiResponse> {
  readonly provider = "cohere" as const;
  private response: any; // Using any to avoid importing Cohere types globally

  constructor(response: any) {
    this.response = response;
  }

  getId(): string {
    return this.response.generationId || "unknown";
  }

  getModel(): string {
    return "cohere-model"; // Cohere responses sometimes don't return the model explicitly
  }

  getText(): string {
    return this.response.text || "";
  }

  getToolCalls(): CommonToolCall[] {
    // Basic tool call mapping if Cohere returns tool_calls (Command R+)
    if (this.response.toolCalls) {
      return this.response.toolCalls.map((tc: any) => ({
        id: "call_" + Math.random().toString(36).substr(2, 9), // Cohere doesn't always give IDs
        name: tc.name,
        arguments: tc.parameters || {},
      }));
    }
    return [];
  }

  hasToolCalls(): boolean {
    return (this.response.toolCalls?.length ?? 0) > 0;
  }

  getUsage(): UsageView {
    return {
      inputTokens: this.response.meta?.tokens?.inputTokens ?? 0,
      outputTokens: this.response.meta?.tokens?.outputTokens ?? 0,
    };
  }

  getOriginalResponse(): OpenAiResponse {
    // We construct a fake OpenAI response to satisfy the interface
    return {
      id: this.getId(),
      object: "chat.completion",
      created: Date.now(),
      model: this.getModel(),
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: this.getText(),
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: this.getUsage().inputTokens,
        completion_tokens: this.getUsage().outputTokens,
        total_tokens: this.getUsage().inputTokens + this.getUsage().outputTokens,
      },
    };
  }

  toRefusalResponse(_refusalMessage: string, contentMessage: string): OpenAiResponse {
    return {
      ...this.getOriginalResponse(),
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: contentMessage,
            refusal: null,
          },
          finish_reason: "stop",
        },
      ],
    };
  }
}

// =============================================================================
// STREAM ADAPTER
// =============================================================================

class CohereStreamAdapter implements LLMStreamAdapter<OpenAiStreamChunk, OpenAiResponse> {
  readonly provider = "cohere" as const;
  readonly state: StreamAccumulatorState;

  constructor() {
    this.state = {
      responseId: "",
      model: "",
      text: "",
      toolCalls: [],
      rawToolCallEvents: [],
      usage: null,
      stopReason: null,
      timing: {
        startTime: Date.now(),
        firstChunkTime: null,
      },
    };
  }

  processChunk(chunk: any): ChunkProcessingResult {
    if (this.state.timing.firstChunkTime === null) {
      this.state.timing.firstChunkTime = Date.now();
    }

    let sseData: string | null = null;
    let isFinal = false;

    // Cohere Stream Events
    if (chunk.eventType === "text-generation") {
      this.state.text += chunk.text;
      const openaiChunk = this.formatToOpenAIChunk(chunk.text);
      sseData = `data: ${JSON.stringify(openaiChunk)}\n\n`;
    } else if (chunk.eventType === "stream-end") {
      isFinal = true;
      if (chunk.response?.meta?.tokens) {
        this.state.usage = {
          inputTokens: chunk.response.meta.tokens.inputTokens,
          outputTokens: chunk.response.meta.tokens.outputTokens,
        };
      }
      this.state.stopReason = chunk.finishReason === "COMPLETE" ? "stop" : "length";
    }

    return { sseData, isToolCallChunk: false, isFinal };
  }

  private formatToOpenAIChunk(content: string): OpenAiStreamChunk {
    return {
      id: this.state.responseId,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: this.state.model,
      choices: [
        {
          index: 0,
          delta: { content },
          finish_reason: null,
        },
      ],
    };
  }

  getSSEHeaders(): Record<string, string> {
    return {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    };
  }

  formatTextDeltaSSE(text: string): string {
    const chunk = this.formatToOpenAIChunk(text);
    return `data: ${JSON.stringify(chunk)}\n\n`;
  }

  getRawToolCallEvents(): string[] {
    return [];
  }

  formatCompleteTextSSE(text: string): string[] {
    return [this.formatTextDeltaSSE(text)];
  }

  formatEndSSE(): string {
    return `data: [DONE]\n\n`;
  }

  toProviderResponse(): OpenAiResponse {
    return {
      id: this.state.responseId,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: this.state.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: this.state.text || null,
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: this.state.usage?.inputTokens ?? 0,
        completion_tokens: this.state.usage?.outputTokens ?? 0,
        total_tokens: (this.state.usage?.inputTokens ?? 0) + (this.state.usage?.outputTokens ?? 0),
      },
    };
  }
}

// =============================================================================
// ADAPTER FACTORY
// =============================================================================

export const cohereAdapterFactory: LLMProvider<
  OpenAiRequest,
  OpenAiResponse,
  OpenAiMessages,
  OpenAiStreamChunk,
  OpenAiHeaders
> = {
  provider: "cohere" as any, // Cast to any if 'cohere' isn't in the types yet
  interactionType: "openai:chatCompletions",

  createRequestAdapter(request: OpenAiRequest): LLMRequestAdapter<OpenAiRequest, OpenAiMessages> {
    return new CohereRequestAdapter(request);
  },

  createResponseAdapter(response: OpenAiResponse): LLMResponseAdapter<OpenAiResponse> {
    return new CohereResponseAdapter(response);
  },

  createStreamAdapter(): LLMStreamAdapter<OpenAiStreamChunk, OpenAiResponse> {
    return new CohereStreamAdapter();
  },

  extractApiKey(headers: OpenAiHeaders): string | undefined {
    return headers.authorization;
  },

  getBaseUrl(): string | undefined {
    return "https://api.cohere.com/v1";
  },

  getSpanName(): string {
    return "cohere.chat";
  },

  createClient(apiKey: string | undefined): CohereClient {
    return new CohereClient({
      token: apiKey || "dummy", // Archestra should provide the key via headers usually
    });
  },

  async execute(client: unknown, request: OpenAiRequest): Promise<OpenAiResponse> {
    const cohereClient = client as CohereClient;
    const { message, chatHistory } = convertMessagesToCohere(request.messages);

    const response = await cohereClient.chat({
      message,
      chatHistory,
      model: request.model,
      temperature: request.temperature ?? 0.7,
    });

    return new CohereResponseAdapter(response).getOriginalResponse();
  },

  async executeStream(client: unknown, request: OpenAiRequest): Promise<AsyncIterable<OpenAiStreamChunk>> {
    const cohereClient = client as CohereClient;
    const { message, chatHistory } = convertMessagesToCohere(request.messages);

    const stream = await cohereClient.chatStream({
      message,
      chatHistory,
      model: request.model,
      temperature: request.temperature ?? 0.7,
    });

    return {
      [Symbol.asyncIterator]: async function* () {
        for await (const chunk of stream) {
          // We yield the raw chunk, the adapter processes it
          yield chunk as any;
        }
      },
    };
  },

  extractErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    return "Unknown Cohere error";
  },
};
