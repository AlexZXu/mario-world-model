#include "command/CommandScheduler.h"

#include <algorithm>

CommandScheduler CommandScheduler::instance;

CommandScheduler& CommandScheduler::getInstance() {
   return instance;
}

void CommandScheduler::addCommand(Command* command) {
   commandQueue.emplace_back(std::move(command));
}

void CommandScheduler::run() {
   const auto count = commandQueue.size();
   for (size_t i = 0; i < count; ++i) {
      auto* command = commandQueue[i];
      command->execute();

      if (command->isFinished()) {
         destroyQueue.push_back(command);
      }
   }

   emptyDestroyQueue();
}

void CommandScheduler::emptyDestroyQueue() {
   if (destroyQueue.empty()) {
      return;
   }

   for (Command* command : destroyQueue) {
      commandQueue.erase(std::remove_if(commandQueue.begin(), commandQueue.end(),
                                        [command](Command* other) {
                                           return command == other;
                                        }),
                         commandQueue.end());

      delete command;
   }

   destroyQueue.clear();
   commandQueue.shrink_to_fit();
}

void CommandScheduler::clear() {
   for (auto* c : commandQueue) delete c;
   commandQueue.clear();
   destroyQueue.clear();
}
