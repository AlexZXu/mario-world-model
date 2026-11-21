#pragma once

#include "ECS/ECS.h"

#include <SDL.h>
#include <SDL_ttf.h>

#include <memory>

enum class BackgroundColor
{
   NONE,
   BLACK,
   BLUE
};

class TextureManager {
  public:
   static TextureManager& Get() {
      return instance;
   }

   int Init();
   int Quit();

   SDL_Texture* LoadTexture(const char* path);
   std::shared_ptr<SDL_Texture> LoadSharedTexture(const char* path, bool blueTransparent = true);
   std::shared_ptr<TTF_Font> LoadSharedFont(const char* path, int fontSize);
   void Draw(SDL_Texture* texture, SDL_Rect destRect);
   void Draw(SDL_Texture* texture, SDL_Rect sourceRect, SDL_Rect destRect);
   void Draw(std::shared_ptr<SDL_Texture>, SDL_Rect sourceRect, SDL_Rect destRect, bool horizontal,
             bool vertical);
   void Draw(std::shared_ptr<TTF_Font> font, const char* text, SDL_Rect position);
   void DrawHorizontalFlipped(std::shared_ptr<SDL_Texture>, SDL_Rect sourceRect, SDL_Rect destRect);
   void DrawVerticalFlipped(std::shared_ptr<SDL_Texture>, SDL_Rect sourceRect, SDL_Rect destRect);
   void SetBackgroundColor(BackgroundColor color);
   void Clear();
   void Display();
   bool SaveFrame(const char* path);
   SDL_Surface* frame = nullptr;

   void ResizeWindow();

   SDL_Renderer* getRenderer() {
      return renderer;
   }

   BackgroundColor getBackgroundColor() {
      return currentColor;
   }

  private:
   TextureManager() {}

   TextureManager(const TextureManager&) = delete;

   static TextureManager instance;

   SDL_Window* window = nullptr;
   SDL_Surface* softwareSurface = nullptr;
   SDL_Renderer* renderer = nullptr;
   BackgroundColor currentColor;
};
