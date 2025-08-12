async def setup(bot):
    # Import lazily so tests that import submodules don't require discord/redbot
    from .cog import GPT5Assistant
    await bot.add_cog(GPT5Assistant(bot))
