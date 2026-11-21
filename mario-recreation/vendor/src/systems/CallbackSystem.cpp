#include "systems/CallbackSystem.h"

#include "ECS/Components.h"
#include "ECS/ECS.h"

void CallbackSystem::tick(World* world) {
   world->find<CallbackComponent>([](Entity* entity) {
      auto* callback = entity->getComponent<CallbackComponent>();

      callback->time--;

      if (callback->time <= 0) {
         // Callbacks may remove/replace their own component. Own the callable
         // during execution and remove the old one before it schedules a new one.
         auto function = callback->callback;
         entity->remove<CallbackComponent>();
         function(entity);
      }
   });

   world->find<WaitUntilComponent>([](Entity* entity) {
      auto* waitUntil = entity->getComponent<WaitUntilComponent>();

      auto condition = waitUntil->condition;
      auto after = waitUntil->doAfter;
      if (condition(entity)) {
         after(entity);
      }
   });

   world->find<TimerComponent>([](Entity* entity) {
      auto* timer = entity->getComponent<TimerComponent>();

      timer->time--;
      if (timer->time == 0) {
         auto function = timer->onExecute;
         timer->time = timer->delay;
         function(entity);
      }
   });

   world->find<DestroyDelayedComponent>([world](Entity* entity) {
      auto* destroy = entity->getComponent<DestroyDelayedComponent>();

      if (destroy->time > 0) {
         destroy->time--;
      } else if (destroy->time <= 0) {
         world->destroy(entity);
      }
   });
}
