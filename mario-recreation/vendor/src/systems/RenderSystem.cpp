#include "BitmapFont.h"
#include "systems/RenderSystem.h"

#include "Camera.h"
#include "Constants.h"
#include "ECS/Components.h"
#include "TextureManager.h"

#include <SDL.h>
#include <SDL_ttf.h>

#include <cmath>
#include <cstring>
#include <iostream>

void RenderSystem::onAddedToWorld(World* world) {
   normalFont = TextureManager::Get().LoadSharedFont("res/fonts/press-start-2p.ttf", 25);
}

void RenderSystem::tick(World* world) {
   TextureManager::Get().Clear();
   // This is to render the entities in the correct order
   if (!transitionRendering) {  // Don't show the entities being loaded during a transition
      world->find<PositionComponent, TextureComponent, BackgroundComponent>([&](Entity* entity) {
         if (Camera::Get().inCameraRange(entity->getComponent<PositionComponent>())) {
            renderEntity(entity);
         }
      });
      world->find<PositionComponent, TextureComponent, ForegroundComponent>([&](Entity* entity) {
         if (Camera::Get().inCameraRange(entity->getComponent<PositionComponent>())) {
            renderEntity(entity);
         }
      });
      world->find<PositionComponent, TextureComponent, ProjectileComponent>([&](Entity* entity) {
         renderEntity(entity);
      });
      world->find<PositionComponent, TextureComponent, CollectibleComponent>([&](Entity* entity) {
         if (Camera::Get().inCameraRange(entity->getComponent<PositionComponent>())) {
            renderEntity(entity);
         }
      });
      world->find<PositionComponent, TextureComponent, EnemyComponent>([&](Entity* entity) {
         if (Camera::Get().inCameraRange(entity->getComponent<PositionComponent>())) {
            renderEntity(entity);
         }
      });
      world->find<PositionComponent, TextComponent, FloatingTextComponent>([&](Entity* entity) {
         renderText(entity, entity->getComponent<TextComponent>()->followCamera);
      });
      world->find<PositionComponent, TextureComponent, PlayerComponent>([&](Entity* entity) {
         renderEntity(entity);
      });
      world->find<PositionComponent, TextureComponent, AboveForegroundComponent>(
          [&](Entity* entity) {
             if (Camera::Get().inCameraRange(entity->getComponent<PositionComponent>())) {
                renderEntity(entity);
             }
          });
      world->find<PositionComponent, TextureComponent, ParticleComponent>([&](Entity* entity) {
         renderEntity(entity);
      });
   }

   world->find<PositionComponent, TextureComponent, IconComponent>([&](Entity* entity) {
      renderEntity(entity, false);
   });
   world->find<PositionComponent, TextComponent>([&](Entity* entity) {
      if (!entity->hasComponent<FloatingTextComponent>()) {
         renderText(entity, entity->getComponent<TextComponent>()->followCamera);
      }
   });

   TextureManager::Get().Display();
}

void RenderSystem::renderEntity(Entity* entity, bool cameraBound) {
   auto* position = entity->getComponent<PositionComponent>();
   auto* texture = entity->getComponent<TextureComponent>();

   if (!texture->isVisible()) {
      return;
   }

   float screenPositionX =
       (cameraBound) ? position->position.x - Camera::Get().getCameraX() : position->position.x;
   float screenPositionY =
       (cameraBound) ? position->position.y - Camera::Get().getCameraY() : position->position.y;

   SDL_Rect destinationRect = {(int)std::floor(screenPositionX / 2) * 2, (int)std::floor(screenPositionY / 2) * 2,
                               position->scale.x, position->scale.y};

   if (entity->hasComponent<SpritesheetComponent>()) {
      auto* spritesheet = entity->getComponent<SpritesheetComponent>();

      TextureManager::Get().Draw(texture->getTexture(), spritesheet->getSourceRect(),
                                 destinationRect, texture->isHorizontalFlipped(),
                                 texture->isVerticalFlipped());
   } else {
      TextureManager::Get().Draw(
          texture->getTexture(), SDL_Rect{0, 0, position->scale.x, position->scale.y},
          destinationRect, texture->isHorizontalFlipped(), texture->isVerticalFlipped());
   }
}

void RenderSystem::renderText(Entity* entity, bool followCamera) {
   auto* p = entity->getComponent<PositionComponent>();
   auto* text = entity->getComponent<TextComponent>();
   if (!text->isVisible()) return;
   int x = int(p->position.x - (followCamera ? Camera::Get().getCameraX() : 0));
   int y = int(p->position.y - (followCamera ? Camera::Get().getCameraY() : 0));
   drawBitmapText(TextureManager::Get().getRenderer(), text->text, x, y, text->fontSize);
}

void RenderSystem::setTransitionRendering(bool transition) {
   transitionRendering = transition;
}
