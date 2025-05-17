import { UserData } from "./types";

const STORAGE_KEY = "zenfill_user_data";

export const storage = {
  async getUserData(): Promise<UserData | null> {
    try {
      const result = await chrome.storage.sync.get(STORAGE_KEY);
      return result[STORAGE_KEY] || null;
    } catch (error) {
      console.error("Error getting user data:", error);
      return null;
    }
  },

  async setUserData(data: UserData): Promise<void> {
    try {
      await chrome.storage.sync.set({ [STORAGE_KEY]: data });
    } catch (error) {
      console.error("Error setting user data:", error);
      throw error;
    }
  },

  async clearUserData(): Promise<void> {
    try {
      await chrome.storage.sync.remove(STORAGE_KEY);
    } catch (error) {
      console.error("Error clearing user data:", error);
      throw error;
    }
  },
};
